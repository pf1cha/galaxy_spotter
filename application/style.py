STYLE_SHEET = """
QMainWindow {
    background-color: #0b0f19;
}
QFrame#Card {
    background-color: #161f30;
    border: 1px solid #233148;
    border-radius: 12px;
}
QLabel {
    color: #f1f5f9;
}
QLabel#HeaderTitle {
    font-size: 20px;
    font-weight: bold;
    color: #f8fafc;
}
QLabel#HeaderSubtitle {
    font-size: 12px;
    color: #94a3b8;
}
QLabel#ResultBadge {
    font-size: 18px;
    font-weight: bold;
    padding: 10px 16px;
    border-radius: 8px;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    font-weight: bold;
    font-size: 13px;
    padding: 10px 18px;
    border-radius: 8px;
    border: none;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton:disabled {
    background-color: #334155;
    color: #64748b;
}
QProgressBar {
    background-color: #0f172a;
    border: 1px solid #233148;
    border-radius: 6px;
    height: 18px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    font-size: 11px;
}
QProgressBar#SpiralBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
    border-radius: 5px;
}
QProgressBar#EllipticalBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d97706, stop:1 #fbbf24);
    border-radius: 5px;
}
"""