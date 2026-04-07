# -*- coding: utf-8 -*-
"""内置计算器工具：安全求值算术表达式。"""

from __future__ import annotations

import ast
import operator as op
from typing import Any

from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter


_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
}


class CalculatorTool(BaseTool):
    """对有限语法树节点求值，禁止任意函数调用。"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "calculator"
        self.description = "计算数学表达式（加减乘除、幂、取模与括号）。"
        self.parameters = [
            ToolParameter(
                name="expression",
                type="string",
                description="仅包含数字与 + - * / ** % 和括号的表达式",
                required=True,
            )
        ]

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            left = self._eval(node.left)
            right = self._eval(node.right)
            return float(_ALLOWED_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return float(_ALLOWED_OPS[type(node.op)](self._eval(node.operand)))
        if isinstance(node, ast.Expr):
            return self._eval(node.value)
        raise ValueError("不支持的表达式语法")

    async def execute(self, **kwargs: Any) -> str:
        """解析并计算表达式字符串。"""
        expr = str(kwargs.get("expression", "")).strip()
        if not expr:
            raise ValueError("参数 expression 不能为空")

        try:
            tree = ast.parse(expr, mode="eval")
            result = self._eval(tree.body)
            logger.info("calculator 计算 {} = {}", expr, result)
            return str(result)
        except Exception as exc:
            logger.warning("calculator 解析失败 expr={} err={}", expr, exc)
            raise ValueError(f"表达式无效或无法计算: {exc!s}") from exc
