"""Settings forms generated from the parameter dataclasses.

Every field of a parameter module (Glider, Rope, ...) gets an editor in its display
unit (unit metadata in params.py).  Tooltips and section headers come from the
comments next to the fields in params.py, so the GUI documents itself and stays in
sync with the model.
"""

import dataclasses
import inspect
import re

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..params import DISPLAY_UNITS

ARRAY_LABELS = ("nose", "wheel", "tail")


def field_docs(cls) -> dict[str, tuple[str, str]]:
    """{field: (comment, section header)} parsed from the class source."""
    try:
        src = inspect.getsource(cls)
    except OSError, TypeError:
        return {}
    # re-join fields the formatter wrapped: `x: T = unit(\n    "N", ...\n)  # doc`
    src = re.sub(r"unit\(\n\s+(.*?),?\n\s+\)", r"unit(\1)", src)
    docs: dict[str, tuple[str, str]] = {}
    section = ""
    prev_comment = False
    for line in src.splitlines():
        m_sec = re.match(r"    # (.+)$", line)
        if m_sec:  # consecutive comment lines form one section header
            text = m_sec.group(1).strip()
            section = f"{section} {text}" if prev_comment else text
            prev_comment = True
            continue
        prev_comment = False
        m = re.match(r"    (\w+): [^=]+=.*?(?:#\s*(.*))?$", line)
        if m and not line.strip().startswith(("def ", '"""')):
            docs[m.group(1)] = ((m.group(2) or "").strip(), section)
    return docs


def fmt(x: float) -> str:
    return f"{x:.6g}"


class FloatEdit(QtWidgets.QLineEdit):
    """Line edit for a float in any magnitude (1e-3 ... 1e12); red when invalid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.setMinimumWidth(90)
        self.setMaximumWidth(120)
        self.textChanged.connect(self._validate)

    def _validate(self):
        ok = self.is_valid()
        self.setStyleSheet("" if ok else "background: #f8d0d0;")

    def is_valid(self) -> bool:
        try:
            float(self.text())
            return True
        except ValueError:
            return False

    def value(self) -> float:
        return float(self.text())

    def set_value(self, x: float):
        self.setText(fmt(float(x)))


class ModuleForm(QtWidgets.QWidget):
    """Editable form for one parameter module; values shown in display units."""

    changed = QtCore.Signal()

    def __init__(self, instance, skip: tuple[str, ...] = (), parent=None):
        super().__init__(parent)
        self.cls = type(instance)
        self.skip = skip
        self.editors: dict[str, list[FloatEdit]] = {}
        self.scale: dict[str, float] = {}
        docs = field_docs(self.cls)

        grid = QtWidgets.QGridLayout(self)
        grid.setColumnStretch(3, 1)
        row = 0
        last_section = None
        for f in dataclasses.fields(self.cls):
            value = getattr(instance, f.name)
            if f.name in skip or dataclasses.is_dataclass(value):
                continue
            doc, section = docs.get(f.name, ("", ""))
            if section and section != last_section:
                head = QtWidgets.QLabel(f"<b>{section}</b>")
                head.setWordWrap(True)
                grid.addWidget(head, row, 0, 1, 4)
                row += 1
                last_section = section
            unit = f.metadata.get("display", f.metadata.get("unit", ""))
            self.scale[f.name] = DISPLAY_UNITS.get(unit, 1.0)
            label = QtWidgets.QLabel(f.name)
            tip = f"{doc}\n[{unit}]" if doc else f"[{unit}]"
            label.setToolTip(tip)
            grid.addWidget(label, row, 0)
            n = np.size(value)
            edits = []
            box = QtWidgets.QHBoxLayout()
            for i in range(n):
                e = FloatEdit()
                e.setToolTip(tip + (f"\n{ARRAY_LABELS[i]}" if n == 3 else ""))
                if n == 3:
                    e.setPlaceholderText(ARRAY_LABELS[i])
                    e.setMinimumWidth(60)
                    e.setMaximumWidth(70)
                e.editingFinished.connect(self.changed)
                box.addWidget(e)
                edits.append(e)
            grid.addLayout(box, row, 1)
            unit_lbl = QtWidgets.QLabel("" if unit == "-" else unit)
            unit_lbl.setStyleSheet("color: gray;")
            grid.addWidget(unit_lbl, row, 2)
            if doc:
                d = QtWidgets.QLabel(doc)
                d.setStyleSheet("color: gray;")
                grid.addWidget(d, row, 3)
            self.editors[f.name] = edits
            row += 1
        grid.setRowStretch(row, 1)
        self.set_instance(instance)

    def set_instance(self, instance):
        for name, edits in self.editors.items():
            vals = np.atleast_1d(np.asarray(getattr(instance, name), float))
            for e, v in zip(edits, vals, strict=True):
                e.set_value(v / self.scale[name])

    def invalid_fields(self) -> list[str]:
        return [
            n for n, es in self.editors.items() if not all(e.is_valid() for e in es)
        ]

    def values_si(self) -> dict:
        """Field values in SI (floats, or numpy arrays for array fields)."""
        out = {}
        for name, edits in self.editors.items():
            vals = [e.value() * self.scale[name] for e in edits]
            out[name] = np.array(vals) if len(vals) > 1 else vals[0]
        return out

    def instance(self, **extra):
        """Build a new parameter module from the form (SI units)."""
        bad = self.invalid_fields()
        if bad:
            raise ValueError(f"{self.cls.__name__}: invalid value for {', '.join(bad)}")
        return self.cls(**self.values_si(), **extra)

    def set_si(self, values: dict):
        for name, v in values.items():
            if name in self.editors:
                vals = np.atleast_1d(np.asarray(v, float))
                for e, x in zip(self.editors[name], vals, strict=True):
                    e.set_value(x / self.scale[name])


def scrolled(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    return area


def small_font() -> QtGui.QFont:
    f = QtGui.QFont()
    f.setPointSizeF(f.pointSizeF() * 0.9)
    return f
