STYLE_SHEET = """
QMainWindow {
    background-color: #100d18;
}
QFrame#Card {
    background-color: #1b1523;
    border: 1px solid #392a43;
    border-radius: 12px;
}
QLabel {
    color: #f7effa;
}
QLabel#HeaderTitle {
    font-size: 22px;
    font-weight: bold;
    color: #fff7ff;
}
QLabel#HeaderSubtitle {
    font-size: 12px;
    color: #b9a7c3;
}
QLabel#ResultBadge {
    font-size: 18px;
    font-weight: bold;
    padding: 10px 16px;
    border-radius: 8px;
}
QPushButton {
    background-color: #8e24aa;
    color: #ffffff;
    font-weight: bold;
    font-size: 13px;
    padding: 10px 18px;
    border-radius: 8px;
    border: 1px solid #a33ac0;
}
QPushButton:hover {
    background-color: #a32bc1;
    border-color: #c64de3;
}
QPushButton:disabled {
    background-color: #33273a;
    color: #897995;
    border-color: #46364f;
}
QProgressBar {
    background-color: #110e19;
    border: 1px solid #392a43;
    border-radius: 6px;
    height: 18px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    font-size: 11px;
}
QProgressBar#SpiralBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8e24aa, stop:1 #d05bea);
    border-radius: 5px;
}
QProgressBar#EllipticalBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e66c12, stop:1 #ffad55);
    border-radius: 5px;
}
"""
