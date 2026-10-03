# -*- coding: utf-8 -*-
"""
代码复杂度分析（Code Complexity Analysis）。

在**编译器已有的产物**上做静态度量，不做任何基于正则的"文本猜测"：
  * 圈复杂度（Cyclomatic Complexity, McCabe 1976）——把代码生成器产出的
    字节码控制流图（CFG）切分为基本块，按 ``E - N + 2`` 计算（函数都是
    单入口单出口，连通分量 P = 1）。条件跳转、短路 ``&& / ||``、循环回边
    都天然体现在字节码的 JUMP/JUMP_IF_* 指令里，因此与实际控制流严格一致。
  * 同时在 **AST** 上独立计算一遍圈复杂度（每个决策点 +1），作为字节码
    结果的交叉校验，两份结果都会返回，前端展示其一致性。
  * 嵌套深度、行数（物理行/有效代码行/注释行/空行）、语句数、参数数、
    调用数等规模指标基于 AST 节点与词法 token 的行号统计。
  * 递归分析基于函数调用图：能识别直接递归与互相（间接）递归，
    并给出所在的递归环。

只消费词法/语法/语义/代码生成流水线的结果，因此「代码到底复不复杂」
得到的是与程序结构一致的量化答案。
"""

from typing import Dict, List, Optional

from . import ast_nodes as ast
from . import bytecode as bc
from . import compiler as compiler_mod


# 圈复杂度风险分级阈值（McCabe 经验区间）
RISK_LOW_MAX = 5        # 1..5  简单，风险低
RISK_MODERATE_MAX = 10  # 6..10 中等复杂，建议关注
RISK_HIGH_MAX = 20      # 11..20 较复杂，建议拆分
# > 20                    高风险，应重构


def risk_level(cc: int) -> str:
    if cc <= RISK_LOW_MAX:
        return "low"
    if cc <= RISK_MODERATE_MAX:
        return "moderate"
    if cc <= RISK_HIGH_MAX:
        return "high"
    return "critical"


RISK_META = {
    "low":      {"label": "简单",   "color": "green", "range": "1–5",
                 "advice": "结构直观，单元测试容易覆盖。"},
    "moderate": {"label": "中等",   "color": "amber", "range": "6–10",
                 "advice": "分支开始增多，建议补充关键路径测试。"},
    "high":     {"label": "复杂",   "color": "red",   "range": "11–20",
                 "advice": "难以完整测试与理解，考虑拆分为更小的函数。"},
    "critical": {"label": "高风险", "color": "red",   "range": ">20",
                 "advice": "圈复杂度过高，重构优先。"},
}


# ===========================================================================
# 入口
# ===========================================================================
def analyze_source(source: str) -> dict:
    """分析一段 MiniLang 源码，返回复杂度报告（dict，可直接 JSON 序列化）。"""
    result = compiler_mod.compile_source(source)
    report = {
        "ok": False,
        "success": result.success,
        "stage": result.stage,
        "diagnostics": result.diagnostics.to_list(),
        "functions": [],
        "summary": None,
    }
    if not result.success or result.ast is None or result.bytecode is None:
        report["ok"] = False
        return report

    analyzer = _Analyzer(result)
    report["functions"] = analyzer.function_reports()
    report["summary"] = analyzer.summary(report["functions"])
    report["ok"] = True
    return report


# ===========================================================================
# 分析器主体
# ===========================================================================
class _Analyzer:
    def __init__(self, result: "compiler_mod.CompileResult"):
        self.result = result
        self.source_lines = result.source_lines
        self.program_ast: ast.Program = result.ast
        self.program_bc = result.bytecode

        self.fn_decls: Dict[str, ast.FunctionDecl] = {}
        for decl in self.program_ast.declarations:
            if isinstance(decl, ast.FunctionDecl):
                self.fn_decls[decl.name] = decl

        # 函数名 -> 定义所在 token 区间（用于物理行数）
        self.fn_spans: Dict[str, tuple] = self._compute_function_spans(result.tokens)
        # 哪些源码行上存在词法记号（注释被 lexer 丢弃，故有记号的行即代码行）
        self.token_lines = self._token_lines(result.tokens)

        # 调用图：函数名 -> 该函数体内调用到的用户函数集合
        self.call_graph: Dict[str, set] = {}
        self._calls_detail: Dict[str, list] = {}
        for name, fn in self.fn_decls.items():
            calls = self._collect_calls(fn.body)
            self.call_graph[name] = {c["callee"] for c in calls if c["callee"] in self.fn_decls}
            self._calls_detail[name] = calls

        # 递归环（SCC，Tarjan）
        self.recursion_info = self._recursion_map()

    # ------------------------------------------------------------------
    # 每个函数的报告
    # ------------------------------------------------------------------
    def function_reports(self) -> List[dict]:
        reports = []
        # 用户自定义函数（按声明顺序）
        for name, fn in self.fn_decls.items():
            fc = self.program_bc.functions.get(name)
            reports.append(self._one_report(name, fn, fc, is_main=False))
        # 顶层 <main>
        if self.program_bc.main is not None:
            main_stmts = [d for d in self.program_ast.declarations if isinstance(d, ast.Stmt)]
            virtual = ast.FunctionDecl("<main>", [], ast.Block(main_stmts), line=1)
            reports.append(self._one_report("<main>", virtual, self.program_bc.main, is_main=True))
        return reports

    def _one_report(self, name, fn: ast.FunctionDecl, fc: Optional[bc.FunctionCode],
                    is_main: bool) -> dict:
        # ---- AST 侧度量 ----
        walker = _AstMetrics(fn.body)
        ast_cc = 1 + walker.decisions
        max_depth = walker.max_depth
        stmt_count = walker.stmts
        decision_points = walker.points
        loops = walker.loops
        branches = walker.branches
        logicals = walker.logicals

        # ---- 行数 ----
        span = self.fn_spans.get(name)
        if is_main:
            # 顶层：取所有顶层语句的首末行
            top_stmts = [d for d in self.program_ast.declarations if isinstance(d, ast.Stmt)]
            if top_stmts:
                span = (min(_node_first_line(s) for s in top_stmts),
                        max(s.line for s in top_stmts))
            else:
                span = (1, 1)
        line_stats = self._line_stats(span)

        # ---- 调用 / 递归 ----
        calls = self._collect_calls(fn.body)
        call_count = len(calls)
        rec = self.recursion_info.get(name, {"recursive": False, "kind": None, "cycle": []})
        direct_self_call = any(c["callee"] == name for c in calls)
        if rec["recursive"]:
            rec["kind"] = "direct" if direct_self_call else "mutual"

        # ---- 字节码侧度量（圈复杂度交叉校验） ----
        cfg = _bytecode_cfg(fc) if fc is not None else None
        bc_cc = cfg["cyclomatic"] if cfg else None
        nodes_n = cfg["nodes"] if cfg else 0
        edges_e = cfg["edges"] if cfg else 0
        blocks = cfg["reachable_blocks"] if cfg else 0
        instr_count = len(fc.instructions) if fc is not None else 0
        cc_match = (bc_cc is not None and bc_cc == ast_cc)

        params = list(fn.params)
        return {
            "name": name,
            "is_main": is_main,
            "location": {"start_line": span[0], "end_line": span[1]},
            "params": params,
            "param_count": len(params),
            # 圈复杂度
            "cyclomatic": ast_cc,
            "cyclomatic_bytecode": bc_cc,
            "cc_match": cc_match,
            "risk": risk_level(ast_cc),
            # 结构指标
            "max_nesting_depth": max_depth,
            "statement_count": stmt_count,
            "decision_count": walker.decisions,
            "branch_count": branches,
            "loop_count": loops,
            "logical_count": logicals,
            "returns": walker.returns,
            # 规模
            "lines": line_stats,
            # 调用与递归
            "call_count": call_count,
            "calls": calls,
            "recursive": rec["recursive"],
            "recursion_kind": rec["kind"],
            "recursion_cycle": rec["cycle"],
            # 字节码 CFG 明细（教学/佐证用）
            "cfg": {
                "blocks": blocks,
                "nodes": nodes_n,
                "edges": edges_e,
                "instructions": instr_count,
                "formula": f"E - N + 2 = {edges_e} - {nodes_n} + 2 = {bc_cc}",
            },
            # 决策点明细（嵌套展开/短路分支可视化）
            "decision_points": decision_points,
        }

    # ------------------------------------------------------------------
    # 整体统计
    # ------------------------------------------------------------------
    def summary(self, fns: List[dict]) -> dict:
        user_fns = [f for f in fns if not f["is_main"]]
        all_cc = [f["cyclomatic"] for f in fns]
        user_cc = [f["cyclomatic"] for f in user_fns] or [0]

        def _dist(vals):
            return {
                "low": sum(1 for v in vals if v <= RISK_LOW_MAX),
                "moderate": sum(1 for v in vals if RISK_LOW_MAX < v <= RISK_MODERATE_MAX),
                "high": sum(1 for v in vals if RISK_MODERATE_MAX < v <= RISK_HIGH_MAX),
                "critical": sum(1 for v in vals if v > RISK_HIGH_MAX),
            }

        total_sloc = sum(f["lines"]["code"] for f in fns)
        total_physical = sum(f["lines"]["total"] for f in fns)
        total_comments = sum(f["lines"]["comment"] for f in fns)
        total_blank = sum(f["lines"]["blank"] for f in fns)
        recursive_fns = [f["name"] for f in fns if f["recursive"]]
        max_fn = max(fns, key=lambda f: f["cyclomatic"], default=None)
        deepest = max(fns, key=lambda f: f["max_nesting_depth"], default=None)
        n = max(1, len(user_fns))
        avg_cc = round(sum(user_cc) / n, 2)
        all_match = all(f["cc_match"] for f in fns)
        return {
            "function_count": len(user_fns),
            "has_main": any(f["is_main"] for f in fns),
            "total_cyclomatic": sum(all_cc),
            "max_cyclomatic": max(all_cc) if all_cc else 0,
            "avg_cyclomatic_user": avg_cc,
            "max_nesting_depth": max((f["max_nesting_depth"] for f in fns), default=0),
            "total_statements": sum(f["statement_count"] for f in fns),
            "total_decisions": sum(f["decision_count"] for f in fns),
            "total_loops": sum(f["loop_count"] for f in fns),
            "total_physical_lines": total_physical,
            "total_code_lines": total_sloc,
            "total_comment_lines": total_comments,
            "total_blank_lines": total_blank,
            "total_instructions": sum(f["cfg"]["instructions"] for f in fns),
            "recursive_count": len(recursive_fns),
            "recursive_functions": recursive_fns,
            "distribution": _dist(user_cc),
            "cc_cross_check_ok": all_match,
            "most_complex": {"name": max_fn["name"], "cyclomatic": max_fn["cyclomatic"]} if max_fn else None,
            "deepest": {"name": deepest["name"], "depth": deepest["max_nesting_depth"]} if deepest else None,
        }

    # ------------------------------------------------------------------
    # 词法 token -> 函数源码区间（{ } 配平）
    # ------------------------------------------------------------------
    def _compute_function_spans(self, tokens) -> Dict[str, tuple]:
        spans = {}
        i = 0
        toks = list(tokens)
        while i < len(toks):
            if toks[i].type == "func":
                name = None
                j = i + 1
                while j < len(toks) and toks[j].type != "{":
                    if toks[j].type == "IDENT" and name is None:
                        name = toks[j].text
                    j += 1
                if j < len(toks) and name is not None:
                    # 从 j 开始做大括号配平
                    depth = 0
                    k = j
                    while k < len(toks):
                        if toks[k].type == "{":
                            depth += 1
                        elif toks[k].type == "}":
                            depth -= 1
                            if depth == 0:
                                spans[name] = (toks[i].line, toks[k].line)
                                i = k
                                break
                        k += 1
            i += 1
        return spans

    # ------------------------------------------------------------------
    # 行分类（基于 lexer 产出的 token；注释识别规则与 lexer 一致）
    # ------------------------------------------------------------------
    def _token_lines(self, tokens) -> set:
        return {t.line for t in tokens if t.type != "EOF"}

    def _line_stats(self, span) -> dict:
        if not span:
            return {"total": 0, "code": 0, "comment": 0, "blank": 0}
        start, end = span
        start = max(1, start)
        end = min(len(self.source_lines), end)
        total = code = comment = blank = 0
        in_block = False
        for ln in range(start, end + 1):
            text = self.source_lines[ln - 1]
            has_code = ln in self.token_lines
            in_block_next, has_comment = self._scan_comment(text, in_block)
            total += 1
            if has_code:
                code += 1
            elif in_block or has_comment:
                comment += 1
            elif not text.strip():
                blank += 1
            else:
                # 行内仅有空白/注释片段（如孤立的 */），归入注释行
                comment += 1
            in_block = in_block_next
        return {"total": total, "code": code, "comment": comment, "blank": blank}

    @staticmethod
    def _scan_comment(text: str, in_block: bool):
        """扫描一行，返回 (扫描后是否处于块注释中, 本行是否经过注释内容)。

        规则与 ``lexer`` 完全一致：``//`` 行注释、``/* ... */`` 块注释；
        字符串内的 // 不算注释——本平台 token 化成功的代码行直接由
        token 集合判定为代码行，这里只处理无 token 的空行/注释行。
        """
        i, n = 0, len(text)
        touched_comment = in_block
        while i < n:
            ch = text[i]
            if in_block:
                touched_comment = True
                if ch == "*" and i + 1 < n and text[i + 1] == "/":
                    in_block = False
                    i += 2
                    continue
                i += 1
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                return in_block, True  # 行注释到行尾，块状态不变
            if ch == "/" and i + 1 < n and text[i + 1] == "*":
                in_block = True
                touched_comment = True
                i += 2
                continue
            i += 1
        return in_block, touched_comment

    # ------------------------------------------------------------------
    # AST 调用收集（含递归调用识别所需的 callee 名）
    # ------------------------------------------------------------------
    def _collect_calls(self, node) -> List[dict]:
        out = []
        self._walk_calls(node, out)
        return out

    def _walk_calls(self, node, out):
        if isinstance(node, ast.CallExpr):
            callee = node.callee.name if isinstance(node.callee, ast.Identifier) else None
            out.append({"callee": callee, "line": node.line,
                        "builtin": not (callee in self.fn_decls)})
        for child in _children(node):
            self._walk_calls(child, out)

    # ------------------------------------------------------------------
    # 递归分析：Tarjan 强连通分量；单点 SCC 且存在自环也算递归
    # ------------------------------------------------------------------
    def _recursion_map(self) -> Dict[str, dict]:
        index_counter = [0]
        stack, lowlink, index, on_stack = [], {}, {}, set()
        sccs: List[List[str]] = []

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)
            for w in self.call_graph.get(v, ()):  # 只连用户函数
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            if lowlink[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)

        import sys
        old_recursionlimit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_recursionlimit, 10000))
        try:
            for v in self.call_graph:
                if v not in index:
                    strongconnect(v)
        finally:
            sys.setrecursionlimit(old_recursionlimit)

        info = {name: {"recursive": False, "kind": None, "cycle": []}
                for name in self.fn_decls}
        for comp in sccs:
            if len(comp) > 1:
                # 互相递归：环按调用关系排序（这里给出 SCC 成员即可）
                cycle = self._order_cycle(comp)
                for name in comp:
                    info[name] = {"recursive": True, "kind": "mutual", "cycle": cycle}
            elif len(comp) == 1:
                v = comp[0]
                if v in self.call_graph.get(v, set()):
                    info[v] = {"recursive": True, "kind": "direct", "cycle": [v, v]}
        return info

    def _order_cycle(self, comp: List[str]) -> List[str]:
        """把互相递归的 SCC 排成一条调用环 A->B->...->A（尽力而为）。"""
        members = set(comp)
        if not members:
            return []
        start = sorted(members)[0]
        ordered = [start]
        cur = start
        visited = {start}
        while True:
            nxts = [w for w in self.call_graph.get(cur, ()) if w in members and w not in visited]
            if not nxts:
                break
            nxt = sorted(nxts)[0]
            ordered.append(nxt)
            visited.add(nxt)
            cur = nxt
        ordered.append(start)
        return ordered


# ===========================================================================
# AST 度量：圈复杂度决策点、最大嵌套深度、语句计数
# ===========================================================================
class _AstMetrics:
    """遍历函数体，累加结构化指标。

    圈复杂度决策点（每个 +1，基数 1）：
      * if / elif 的每个条件分支（``IfStmt.branches`` 每一项）；
      * while / for 各算 1 个循环决策；
      * 逻辑运算符 ``&&`` / ``||``（短路求值等价于额外的条件跳转）。
    注意 ``else`` 本身不增加复杂度（它只是分支的汇合路径）。
    """

    def __init__(self, body: ast.Block):
        self.decisions = 0
        self.max_depth = 0
        self.stmts = 0
        self.loops = 0
        self.branches = 0
        self.logicals = 0
        self.returns = 0
        self.points: List[dict] = []
        if body is not None:
            self._walk_block(body, depth=0)

    # ---- 语句 ----
    def _walk_block(self, block: ast.Block, depth):
        for s in block.statements:
            self._walk_stmt(s, depth)

    def _walk_stmt(self, s, depth):
        if isinstance(s, ast.Block):
            self.stmts += 1
            self._walk_block(s, depth)
        elif isinstance(s, ast.IfStmt):
            self.stmts += 1
            self._walk_if(s, depth)
        elif isinstance(s, ast.WhileStmt):
            self.stmts += 1
            self.loops += 1
            nd = depth + 1
            self.max_depth = max(self.max_depth, nd)
            self._add_point("loop", "while 循环", s.condition.line, nd,
                            "循环回边构成一个判定节点")
            self._walk_expr(s.condition, nd)
            self._walk_block(s.body, nd)
        elif isinstance(s, ast.ForStmt):
            self.stmts += 1
            self.loops += 1
            nd = depth + 1
            self.max_depth = max(self.max_depth, nd)
            self._add_point("loop", "for 循环", s.line, nd,
                            "条件判定 + 回边构成一个判定节点")
            if s.init:
                self._walk_stmt(s.init, depth)  # 初始化不增加嵌套
            if s.condition:
                self._walk_expr(s.condition, nd)
            if s.increment:
                self._walk_expr(s.increment, nd)
            self._walk_block(s.body, nd)
        elif isinstance(s, ast.ReturnStmt):
            self.stmts += 1
            self.returns += 1
            if s.value:
                self._walk_expr(s.value, depth)
        elif isinstance(s, (ast.BreakStmt, ast.ContinueStmt)):
            self.stmts += 1
        elif isinstance(s, ast.VarDecl):
            self.stmts += 1
            if s.initializer:
                self._walk_expr(s.initializer, depth)
        elif isinstance(s, ast.AssignStmt):
            self.stmts += 1
            self._walk_expr(s.target, depth)
            self._walk_expr(s.value, depth)
        elif isinstance(s, ast.ExprStmt):
            self.stmts += 1
            self._walk_expr(s.expr, depth)
        elif isinstance(s, ast.PrintStmt):
            self.stmts += 1
            for a in s.args:
                self._walk_expr(a, depth)

    def _walk_if(self, s: ast.IfStmt, depth):
        nd = depth + 1
        self.max_depth = max(self.max_depth, nd)
        for idx, (cond, body) in enumerate(s.branches):
            kind_word = "if 条件" if idx == 0 else "elif 条件"
            self.branches += 1
            self._add_point("branch", kind_word, cond.line, nd,
                            "条件跳转 JUMP_IF_FALSE 增加一条边")
            self._walk_expr(cond, nd)
            self._walk_block(body, nd)
        if s.else_block:
            # else 不增加圈复杂度，但内部语句的嵌套层级 +1
            self._walk_block(s.else_block, nd)

    # ---- 表达式（只需统计 LogicalExpr） ----
    def _walk_expr(self, e, depth):
        if isinstance(e, ast.LogicalExpr):
            self.logicals += 1
            symbol = "&&" if e.op == "&&" else "||"
            self._add_point("logical", f"逻辑 {symbol}（短路分支）", e.line, depth,
                            "短路求值编译为条件跳转，等价一个判定节点")
            self._walk_expr(e.left, depth)
            self._walk_expr(e.right, depth)
        else:
            for child in _children(e):
                self._walk_expr(child, depth)

    def _add_point(self, kind, label, line, depth, note):
        self.decisions += 1
        self.points.append({
            "kind": kind, "label": label, "line": line,
            "nesting": depth, "note": note,
        })


def _node_first_line(node) -> int:
    line = getattr(node, "line", 1)
    for child in _children(node):
        line = min(line, _node_first_line(child))
    return line


def _children(node):
    """AST 节点的结构性子节点（按各节点的语义字段取，不用 to_dict）。"""
    if node is None:
        return []
    children = []
    if isinstance(node, ast.UnaryExpr):
        children = [node.operand]
    elif isinstance(node, (ast.BinaryExpr, ast.LogicalExpr)):
        children = [node.left, node.right]
    elif isinstance(node, ast.CallExpr):
        children = [node.callee] + list(node.args)
    elif isinstance(node, ast.IndexExpr):
        children = [node.target, node.index]
    elif isinstance(node, ast.ListLiteral):
        children = list(node.elements)
    elif isinstance(node, ast.VarDecl):
        children = [node.initializer] if node.initializer else []
    elif isinstance(node, ast.AssignStmt):
        children = [node.target, node.value]
    elif isinstance(node, ast.ExprStmt):
        children = [node.expr]
    elif isinstance(node, ast.PrintStmt):
        children = list(node.args)
    elif isinstance(node, ast.ReturnStmt):
        children = [node.value] if node.value else []
    elif isinstance(node, ast.WhileStmt):
        children = [node.condition, node.body]
    elif isinstance(node, ast.ForStmt):
        children = [node.init, node.condition, node.increment, node.body]
    elif isinstance(node, ast.Block):
        children = list(node.statements)
    elif isinstance(node, ast.IfStmt):
        for cond, body in node.branches:
            children += [cond, body]
        if node.else_block:
            children.append(node.else_block)
    elif isinstance(node, ast.FunctionDecl):
        children = [node.body]
    return [c for c in children if c is not None]


# ===========================================================================
# 字节码控制流图：基本块划分 + 圈复杂度 E - N + 2
# ===========================================================================
_JUMP_OPS = {bc.OP_JUMP, bc.OP_JUMP_IF_FALSE, bc.OP_JUMP_IF_TRUE}
_TERMINATOR_OPS = _JUMP_OPS | {bc.OP_RETURN, bc.OP_RETURN_NONE}


def _bytecode_cfg(fc: bc.FunctionCode) -> dict:
    """从函数字节码构建 CFG 并计算圈复杂度。

    基本块（basic block）leader 规则（经典算法）：
      1. 第一条指令是 leader；
      2. 任意跳转指令的目标是 leader；
      3. 任意跳转/条件跳转指令的下一条是 leader；
      4. RETURN/RETURN_NONE 的下一条是 leader（块以返回结束）。
    边：
      * 无条件跳转：块末 -> 目标块；
      * 条件跳转：块末 -> 目标块（taken）+ 顺序下一块（fall-through）；
      * 返回指令：连向合成的 exit 节点；若最后一个块不以返回结束，也连 exit；
      * 顺序落入下一块的非终止块：连向下一块。

    圈复杂度 = E - N + 2（McCabe；函数单入口单出口，P=1）。
    回边（while/for/continue）自然出现在边集中，无需特判；break/continue
    在代码生成阶段已回填为普通 JUMP，语义上与结构图完全一致。
    """
    ins = fc.instructions
    n = len(ins)
    if n == 0:
        return {"cyclomatic": 1, "nodes": 1, "edges": 0, "reachable_blocks": 0}

    def _valid(target):
        return isinstance(target, int) and 0 <= target < n

    # 1) 找 leaders
    leaders = {0}
    for i, instr in enumerate(ins):
        if instr.op in _JUMP_OPS and _valid(instr.operand):
            leaders.add(instr.operand)
            if i + 1 < n:
                leaders.add(i + 1)
        elif instr.op in (bc.OP_RETURN, bc.OP_RETURN_NONE):
            if i + 1 < n:
                leaders.add(i + 1)

    leader_sorted = sorted(leaders)
    # leader -> block id
    block_of = {}
    ranges = []
    for bi, start in enumerate(leader_sorted):
        end = leader_sorted[bi + 1] if bi + 1 < len(leader_sorted) else n
        block_of[start] = bi
        ranges.append((start, end))

    # 2) 块内最后一条有效控制流指令
    blocks = []
    for bi, (start, end) in enumerate(ranges):
        last = None
        for k in range(start, end):
            if ins[k].op in _TERMINATOR_OPS:
                last = k
                break
        blocks.append({"id": bi, "start": start, "end": end, "term": last})

    # 3) 建边（去重）
    exit_id = len(blocks)  # 合成 exit 节点
    edges = set()

    def _edge(a, b):
        edges.add((a, b))

    for blk in blocks:
        term = blk["term"]
        if term is None:
            # 无终止指令：顺序连下一块；位于末尾则连 exit
            if blk["id"] + 1 < len(blocks):
                _edge(blk["id"], blk["id"] + 1)
            else:
                _edge(blk["id"], exit_id)
            continue
        instr = ins[term]
        if instr.op == bc.OP_JUMP:
            if _valid(instr.operand):
                _edge(blk["id"], block_of[_leader_at(leaders, instr.operand)])
        elif instr.op in (bc.OP_JUMP_IF_FALSE, bc.OP_JUMP_IF_TRUE):
            # taken
            if _valid(instr.operand):
                _edge(blk["id"], block_of[_leader_at(leaders, instr.operand)])
            # fall-through：终止指令的下一条所在块（通常是本块后续的新块）
            if term + 1 < n:
                _edge(blk["id"], block_of[_leader_at(leaders, term + 1)])
            else:
                _edge(blk["id"], exit_id)
        elif instr.op in (bc.OP_RETURN, bc.OP_RETURN_NONE):
            _edge(blk["id"], exit_id)
            # 返回之后若还有可达指令（下一个 leader），不作为结构边连入

    # 4) 从入口做可达性分析（只统计真实可达的块）
    adj: Dict[int, list] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v != exit_id and v not in seen:
                seen.add(v)
                stack.append(v)

    reachable_blocks = len(seen)
    reachable_edges = [(a, b) for (a, b) in edges
                       if a in seen and (b == exit_id or b in seen)]
    # N 含合成 exit 节点（exit 从入口可达）
    nodes = reachable_blocks + 1
    e = len(reachable_edges)
    cc = e - nodes + 2
    return {
        "cyclomatic": cc,
        "nodes": nodes,
        "edges": e,
        "reachable_blocks": reachable_blocks,
    }


def _leader_at(leaders, target) -> int:
    """跳转目标所在基本块的 leader（目标本身就是 leader，防御性地向前兜底）。"""
    if target in leaders:
        return target
    before = [l for l in leaders if l <= target]
    return max(before) if before else 0
