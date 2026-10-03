# -*- coding: utf-8 -*-
"""
代码复杂度分析。

指标**直接建立在现有编译产物之上**，不重新解析源码：
  * 圈复杂度（McCabe Cyclomatic Complexity）——由代码生成器产出的字节码构建
    控制流图（CFG）：每条 ``JUMP_IF_FALSE`` / ``JUMP_IF_TRUE`` 都是一个分支谓词
    （if/elif 条件、while/for 条件，以及 ``&&`` / ``||`` 的短路跳转），因此
    ``v(G) = 1 + 谓词数``，同时用经典公式 ``E - N + 2``（补一个虚拟出口节点）
    交叉验证，两个数值必须一致；
  * 嵌套深度、语句数、判定点明细——遍历语义分析后的 AST 结构得到，与 CFG 的
    谓词数按 ``if/elif 分支 + 循环 + 逻辑短路`` 分项对账；
  * 行数——用 AST 行号确定每个函数（含顶层 <main>）的源码区间，统计物理行、
    有效代码行（SLOC，去空行/注释）以及字节码实际覆盖的可执行行；
  * 调用与递归——扫描字节码里的 ``LOAD_FUNC … CALL`` 对得到调用关系，再用
    Tarjan 强连通分量识别直接递归与相互递归。
"""

from typing import Dict, List, Optional

from . import ast_nodes as ast
from . import bytecode as bc


# 条件跳转（CFG 谓词）
_COND_JUMPS = {bc.OP_JUMP_IF_FALSE, bc.OP_JUMP_IF_TRUE}
# McCabe 圈复杂度风险分级
RATING = [
    (10, "high", "高风险"),
    (5, "medium", "中等"),
    (1, "low", "简单"),
]


def rate(cyclomatic: int):
    for threshold, key, label in RATING:
        if cyclomatic >= threshold:
            return key, label
    return "low", "简单"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def analyze(result) -> dict:
    """对一次编译产物（CompileResult）做复杂度分析。

    成功编译（有 AST 与字节码）时返回完整指标；否则返回 success=False 由页面
    引导用户查看诊断。
    """
    if result.ast is None or result.bytecode is None:
        return {"ok": False, "functions": [], "summary": None}

    source_lines = result.source_lines or result.source.split("\n")
    program = result.ast
    bytecode = result.bytecode

    # 源码区间：函数按定义出现顺序，顶层 <main> 放在最后与字节码页签保持一致
    fn_decls: Dict[str, ast.FunctionDecl] = {}
    fn_order: List[str] = []
    main_stmts: List[ast.Stmt] = []
    for decl in program.declarations:
        if isinstance(decl, ast.FunctionDecl):
            fn_decls[decl.name] = decl
            fn_order.append(decl.name)
        elif isinstance(decl, ast.Stmt):
            main_stmts.append(decl)

    segments = []  # (名称, AST 根节点列表, FunctionCode)
    for name in fn_order:
        fc = bytecode.functions.get(name)
        if fc is not None:
            segments.append((name, [fn_decls[name]], fc))
    if bytecode.main is not None:
        segments.append(("<main>", main_stmts, bytecode.main))

    functions = []
    for name, roots, fc in segments:
        functions.append(_analyze_segment(name, roots, fc, source_lines))

    _analyze_recursion(functions)
    summary = _build_summary(functions, source_lines)
    return {"ok": True, "functions": functions, "summary": summary}


# ---------------------------------------------------------------------------
# 单个函数（含 <main>）
# ---------------------------------------------------------------------------
def _analyze_segment(name, roots, fc, source_lines) -> dict:
    struct = _StructWalker()
    for root in roots:
        struct.visit(root)

    cfg = _cfg_metrics(fc)
    # 调用图只需要知道「调用了谁」：LOAD_FUNC 的操作数一定是已注册用户函数，
    # LOAD_BUILTIN（print/len/...）不会产生用户函数调用边。
    calls = _extract_calls(fc)
    cyclomatic = cfg["cyclomatic_predicates"]
    level, level_label = rate(cyclomatic)

    start_line, end_line = _span(roots)
    physical, blank, comment = _count_lines(start_line, end_line, source_lines)
    code_lines = physical - blank - comment
    if roots:
        exec_lines = len({ins.line for ins in fc.instructions
                          if start_line <= ins.line <= end_line})
    else:
        # 空 <main>：末尾的 RETURN_NONE 是编译器合成指令（line=1），不算可执行行
        exec_lines = 0

    # 判定点分项（AST 结构）应与字节码里的条件跳转数逐一对账
    pred_breakdown = {
        "branch": struct.if_branches,
        "loop": struct.loops,
        "logical": struct.logical_ops,
    }
    ast_predicates = sum(pred_breakdown.values())

    return {
        "name": name,
        "is_main": fc.is_main,
        "params": list(fc.params),
        "arity": fc.arity,
        # ---- 圈复杂度（两种算法交叉验证）----
        "cyclomatic": cyclomatic,
        "cyclomatic_cfg": cfg["cyclomatic_formula"],
        "rating": level,
        "rating_label": level_label,
        "predicate_count": cfg["predicates"],
        "predicate_breakdown": pred_breakdown,
        "metrics_consistent": cyclomatic == cfg["cyclomatic_formula"]
                              and ast_predicates == cfg["predicates"],
        "cfg_nodes": cfg["nodes"],
        "cfg_edges": cfg["edges"],
        "cfg_blocks": cfg["blocks"],
        # ---- 嵌套 / 语句 ----
        "max_nesting": struct.max_depth,
        "max_nesting_line": struct.max_depth_line,
        "statement_count": struct.statements,
        # ---- 行数 ----
        "start_line": start_line,
        "end_line": end_line,
        "physical_lines": physical,
        "code_lines": code_lines,
        "blank_lines": blank,
        "comment_lines": comment,
        "executable_lines": exec_lines,
        "instruction_count": len(fc.instructions),
        # ---- 控制结构统计 ----
        "if_statements": struct.if_statements,
        "if_branches": struct.if_branches,
        "loop_count": struct.loops,
        "logical_count": struct.logical_ops,
        "break_count": struct.breaks,
        "continue_count": struct.continues,
        "return_count": struct.returns,
        "decisions": struct.decisions,
        # ---- 调用 / 递归 ----
        "calls": calls,
        "call_sites": sum(c["count"] for c in calls),
        "recursive": False,
        "recursion_type": "none",
        "recursion_cycle": [],
    }


def _span(roots):
    """返回 (起始行, 结束行)：取所有节点行号的最小/最大值。"""
    if not roots:
        return 1, 1
    lines = []

    def walk(node):
        if isinstance(node, ast.Node):
            lines.append(node.line)
            for child in _children(node):
                walk(child)

    for r in roots:
        walk(r)
    return (min(lines), max(lines)) if lines else (1, 1)


def _children(node) -> list:
    """AST 节点的直接子节点（与具体字段对应，避免依赖 to_dict）。"""
    out = []
    if isinstance(node, ast.UnaryExpr):
        out.append(node.operand)
    elif isinstance(node, (ast.BinaryExpr, ast.LogicalExpr)):
        out.extend([node.left, node.right])
    elif isinstance(node, ast.CallExpr):
        out.append(node.callee)
        out.extend(node.args)
    elif isinstance(node, ast.IndexExpr):
        out.extend([node.target, node.index])
    elif isinstance(node, ast.ListLiteral):
        out.extend(node.elements)
    elif isinstance(node, ast.VarDecl):
        if node.initializer:
            out.append(node.initializer)
    elif isinstance(node, ast.AssignStmt):
        out.extend([node.target, node.value])
    elif isinstance(node, ast.ExprStmt):
        out.append(node.expr)
    elif isinstance(node, ast.PrintStmt):
        out.extend(node.args)
    elif isinstance(node, ast.ReturnStmt):
        if node.value:
            out.append(node.value)
    elif isinstance(node, ast.Block):
        out.extend(node.statements)
    elif isinstance(node, ast.IfStmt):
        for cond, body in node.branches:
            out.extend([cond, body])
        if node.else_block:
            out.append(node.else_block)
    elif isinstance(node, ast.WhileStmt):
        out.extend([node.condition, node.body])
    elif isinstance(node, ast.ForStmt):
        if node.init:
            out.append(node.init)
        if node.condition:
            out.append(node.condition)
        if node.increment:
            out.append(node.increment)
        out.append(node.body)
    elif isinstance(node, ast.FunctionDecl):
        out.append(node.body)
    elif isinstance(node, ast.Program):
        out.extend(node.declarations)
    return [c for c in out if c is not None]


def _count_lines(start, end, source_lines) -> tuple:
    """统计 [start, end] 行区间内的物理行 / 空行 / 纯注释行（1-based，闭区间）。"""
    physical = blank = comment = 0
    for i in range(start - 1, min(end, len(source_lines))):
        physical += 1
        text = source_lines[i].strip()
        if not text:
            blank += 1
        elif text.startswith("//") or text.startswith("/*") or text.startswith("*"):
            comment += 1
    return physical, blank, comment


# ---------------------------------------------------------------------------
# AST 结构遍历：嵌套深度 / 语句数 / 判定点
# ---------------------------------------------------------------------------
class _StructWalker:
    def __init__(self):
        self.statements = 0
        self.if_statements = 0
        self.if_branches = 0
        self.loops = 0
        self.logical_ops = 0
        self.breaks = 0
        self.continues = 0
        self.returns = 0
        self.max_depth = 0
        self.max_depth_line = 0
        self.decisions: List[dict] = []

    def visit(self, node, depth=0):
        # 语句计数：Block 只是分组容器，不计入；函数体本身是 Block 也不计数
        if isinstance(node, ast.Stmt) and not isinstance(node, ast.Block):
            self.statements += 1

        if isinstance(node, ast.IfStmt):
            self.if_statements += 1
            for idx, (cond, body) in enumerate(node.branches):
                self.if_branches += 1
                self._enter_control(depth + 1, cond.line,
                                    "if 分支条件" if idx == 0 else "elif 分支条件")
                self.visit(cond, depth)
                self.visit(body, depth + 1)
            if node.else_block:
                self.visit(node.else_block, depth + 1)
            return

        if isinstance(node, ast.WhileStmt):
            self.loops += 1
            self._enter_control(depth + 1, node.line, "while 循环条件")
            self.visit(node.condition, depth)
            self.visit(node.body, depth + 1)
            return

        if isinstance(node, ast.ForStmt):
            self.loops += 1
            self._enter_control(depth + 1, node.line, "for 循环条件")
            if node.init:
                self.visit(node.init, depth)
            if node.condition:
                self.visit(node.condition, depth)
            if node.increment:
                self.visit(node.increment, depth)
            self.visit(node.body, depth + 1)
            return

        if isinstance(node, ast.LogicalExpr):
            self.logical_ops += 1
            label = "&& 短路分支" if node.op == "&&" else "|| 短路分支"
            self._decision(node.line, "logical", label, node.op)
            self.visit(node.left, depth)
            self.visit(node.right, depth)
            return

        if isinstance(node, ast.BreakStmt):
            self.breaks += 1
        elif isinstance(node, ast.ContinueStmt):
            self.continues += 1
        elif isinstance(node, ast.ReturnStmt):
            self.returns += 1

        for child in _children(node):
            self.visit(child, depth)

    def _enter_control(self, depth, line, label):
        self._decision(line, "branch" if "分支" in label else "loop", label, "")
        if depth > self.max_depth:
            self.max_depth = depth
            self.max_depth_line = line

    def _decision(self, line, kind, label, op):
        self.decisions.append({"line": line, "kind": kind, "label": label,
                               "op": op})
        self.decisions.sort(key=lambda d: (d["line"], d["kind"]))


# ---------------------------------------------------------------------------
# 字节码控制流图：圈复杂度
# ---------------------------------------------------------------------------
def _cfg_metrics(fc: bc.FunctionCode) -> dict:
    """从字节码构建 CFG。

    基本块按「首领（leader）」切分：第一条指令、跳转目标、条件/无条件跳转的
    下一条指令。补一个虚拟 EXIT 节点后，McCabe 公式 ``E - N + 2`` 与
    ``1 + 条件跳转数`` 恒等，二者同时返回用于对账。

    跳转操作数在代码生成时是指令列表的 0-based 下标（展示用 offset 字段是
    1-based，仅服务 UI）；越界目标（理论上不应出现）按导向 EXIT 处理，保证
    分析本身永远不抛异常。
    """
    ins = fc.instructions
    n = len(ins)
    if n == 0:
        return {"blocks": 0, "nodes": 1, "edges": 0, "predicates": 0,
                "cyclomatic_predicates": 1, "cyclomatic_formula": 1}

    leaders = {0}
    for idx, i in enumerate(ins):
        if i.op in _COND_JUMPS or i.op == bc.OP_JUMP:
            tgt = i.operand
            if isinstance(tgt, int) and 0 <= tgt < n:
                leaders.add(tgt)
            if idx + 1 < n:
                leaders.add(idx + 1)
        elif i.op in (bc.OP_RETURN, bc.OP_RETURN_NONE) and idx + 1 < n:
            leaders.add(idx + 1)

    leader_list = sorted(leaders)
    block_of: Dict[int, int] = {}
    for b_idx, lead in enumerate(leader_list):
        nxt = leader_list[b_idx + 1] if b_idx + 1 < len(leader_list) else n
        for k in range(lead, nxt):
            block_of[k] = b_idx
    block_count = len(leader_list)
    exit_node = block_count  # 虚拟 EXIT 的节点号

    edges = set()
    predicates = 0
    for idx, i in enumerate(ins):
        b = block_of[idx]
        is_last = (idx + 1 >= n) or (idx + 1 in block_of and block_of[idx + 1] != b)
        if not is_last:
            continue
        if i.op in _COND_JUMPS:
            predicates += 1
            # 真/假两条边：跳转目标 + 顺序下一条
            tgt = i.operand
            edges.add((b, block_of[tgt] if isinstance(tgt, int) and 0 <= tgt < n else exit_node))
            edges.add((b, block_of[idx + 1] if idx + 1 < n else exit_node))
        elif i.op == bc.OP_JUMP:
            tgt = i.operand
            edges.add((b, block_of[tgt] if isinstance(tgt, int) and 0 <= tgt < n else exit_node))
        elif i.op in (bc.OP_RETURN, bc.OP_RETURN_NONE):
            edges.add((b, exit_node))
        else:
            edges.add((b, block_of[idx + 1] if idx + 1 < n else exit_node))

    # 可达性（跳转折叠可能留下不可达块）
    succ: Dict[int, set] = {b: set() for b in range(block_count + 1)}
    for a, b2 in edges:
        succ[a].add(b2)
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for to in succ[cur]:
            if to not in seen:
                seen.add(to)
                stack.append(to)
    reachable_edges = {(a, b2) for a, b2 in edges if a in seen and b2 in seen}
    reachable_nodes = len(seen)

    cyc_formula = len(reachable_edges) - reachable_nodes + 2
    return {
        "blocks": block_count,
        "nodes": reachable_nodes,
        "edges": len(reachable_edges),
        "predicates": predicates,
        "cyclomatic_predicates": 1 + predicates,
        "cyclomatic_formula": cyc_formula,
    }


# ---------------------------------------------------------------------------
# 调用关系 / 递归
# ---------------------------------------------------------------------------
def _extract_calls(fc: bc.FunctionCode) -> list:
    """扫描字节码找出对**用户函数**的调用点。

    LOAD_FUNC <name> 一定与配对的 CALL 相邻（参数压栈在二者之间）。精确模拟
    栈式虚拟机的净栈效应即可稳健配对，嵌套调用（如 f(g(1))）同样成立；
    LOAD_BUILTIN（print/len/...）不是用户函数，不计入调用图。
    """
    stack: List[Optional[str]] = []
    calls: Dict[str, dict] = {}

    def add(name, line):
        e = calls.setdefault(name, {"name": name, "count": 0, "lines": []})
        e["count"] += 1
        if line not in e["lines"]:
            e["lines"].append(line)

    def pop(k=1):
        for _ in range(k):
            if stack:
                stack.pop()

    for ins in fc.instructions:
        if ins.op == bc.OP_LOAD_FUNC:
            stack.append(ins.operand)
        elif ins.op in (bc.OP_LOAD_CONST, bc.OP_LOAD_VAR, bc.OP_LOAD_GLOBAL,
                        bc.OP_LOAD_BUILTIN):
            stack.append(None)
        elif ins.op == bc.OP_UNARY:
            pop(1)
            stack.append(None)
        elif ins.op == bc.OP_BINARY or ins.op == bc.OP_INDEX_LOAD:
            pop(2)
            stack.append(None)
        elif ins.op == bc.OP_MAKE_LIST:
            pop(ins.operand or 0)
            stack.append(None)
        elif ins.op == bc.OP_DUP:
            stack.append(stack[-1] if stack else None)
        elif ins.op == bc.OP_DUP2:
            if len(stack) >= 2:
                stack.extend([stack[-2], stack[-1]])
            else:
                stack.extend([None, None])
        elif ins.op == bc.OP_CALL:
            pop(ins.operand or 0)              # 参数
            func_slot = stack.pop() if stack else None
            if func_slot:
                add(func_slot, ins.line)
            stack.append(None)                # 调用结果留在栈上（供嵌套调用）
        elif ins.op in (bc.OP_POP, bc.OP_STORE_VAR, bc.OP_STORE_GLOBAL):
            pop(1)
        elif ins.op == bc.OP_INDEX_STORE:
            pop(3)
    return [calls[k] for k in sorted(calls)]


def _analyze_recursion(functions: list):
    """Tarjan 强连通分量：自环为直接递归，多节点环为相互递归。"""
    index_of: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    counter = [0]
    sccs: List[List[str]] = []
    by_name = {f["name"]: f for f in functions}

    graph = {f["name"]: {c["name"] for c in f["calls"] if c["name"] in by_name}
             for f in functions}

    def strongconnect(v):
        index_of[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in graph[v]:
            if w not in index_of:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], index_of[w])
        if low[v] == index_of[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in graph:
        if v not in index_of:
            strongconnect(v)

    for comp in sccs:
        if len(comp) == 1:
            v = comp[0]
            if v in graph[v]:
                by_name[v]["recursive"] = True
                by_name[v]["recursion_type"] = "direct"
                by_name[v]["recursion_cycle"] = [v]
        else:
            for v in comp:
                by_name[v]["recursive"] = True
                by_name[v]["recursion_type"] = "mutual"
                by_name[v]["recursion_cycle"] = sorted(comp)


# ---------------------------------------------------------------------------
# 整体统计
# ---------------------------------------------------------------------------
def _build_summary(functions: list, source_lines: list) -> dict:
    user_fns = [f for f in functions if not f["is_main"]]
    cyc_values = [f["cyclomatic"] for f in functions] or [1]
    nesting_values = [f["max_nesting"] for f in functions] or [0]
    total_lines = len(source_lines)
    total_code = sum(f["code_lines"] for f in user_fns)
    total_physical = sum(f["physical_lines"] for f in user_fns)
    recursive_fns = [f["name"] for f in functions if f["recursive"]]
    decisions_total = sum(f["predicate_count"] for f in functions)

    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0

    dist = {"low": 0, "medium": 0, "high": 0}
    for f in functions:
        dist[f["rating"]] += 1

    return {
        "function_count": len(user_fns),
        "segment_count": len(functions),
        "total_source_lines": total_lines,
        "total_code_lines": sum(f["code_lines"] for f in functions),
        "total_function_physical_lines": total_physical,
        "total_function_code_lines": total_code,
        "total_statements": sum(f["statement_count"] for f in functions),
        "total_instructions": sum(f["instruction_count"] for f in functions),
        "total_decisions": decisions_total,
        "cyclomatic_total": sum(cyc_values),
        "cyclomatic_max": max(cyc_values),
        "cyclomatic_avg": _avg(cyc_values),
        "max_nesting": max(nesting_values),
        "recursive_count": len(recursive_fns),
        "recursive_functions": recursive_fns,
        "rating_distribution": dist,
        "all_consistent": all(f["metrics_consistent"] for f in functions),
    }
