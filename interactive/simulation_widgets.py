"""Readable exact numbers and observable text for the Simulator UI."""

from contextlib import suppress

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QLineEdit


class NumberEdit(QLineEdit):
    """Compact presentation; focus and commits retain the complete input string."""

    activated = Signal()
    committed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._exact = ""
        self.editingFinished.connect(self._finish)
        self.setMinimumWidth(100)

    def exact_text(self):
        return self.text() if self.isModified() else self._exact

    def setText(self, text):
        if self.hasFocus() and self.isModified():
            return
        self._exact = str(text)
        self.setToolTip("Exact value: " + self._exact)
        self._present()

    def _present(self):
        text = self._exact
        if not self.hasFocus():
            with suppress(ValueError):
                text = format(float(text), ".6g")
        super().setText(text)
        self.setModified(False)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.activated.emit()
        self._present()
        self.selectAll()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._present()

    def _finish(self):
        if self.isModified():
            self._exact = self.text()
            self.setModified(False)
            self.committed.emit(self._exact)
        self.setToolTip("Exact value: " + self._exact)
        self._present()


class StatusLabel(QLabel):
    changed = Signal()

    def setText(self, text):
        super().setText(text)
        self.changed.emit()
