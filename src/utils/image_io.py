# src/utils/image_io.py
"""Robust image loading helpers.

Some real-world datasets ship files whose content does not match their file
extension - e.g. JPEG bytes stored as ``foo.png`` (the TXL-PBC dataset has
~92% of its images like this).

Qt's ``QImageReader`` picks the decoder from the **file extension** first, so
such files fail to decode: ``QPixmap(path)`` returns a null pixmap and the
annotation canvas stays blank even though the file is a perfectly valid JPEG.

``load_qpixmap()`` asks Qt to decide the format from the file **content**
instead (``setDecideFormatFromContent``), which loads those files correctly
while behaving exactly like before for well-named files.
"""
from PyQt5.QtGui import QImageReader, QPixmap


def load_qpixmap(path):
    """Load ``path`` into a QPixmap, sniffing the real format from the content.

    Returns a *null* QPixmap when the file cannot be decoded (callers already
    guard on ``pixmap.isNull()``).
    """
    try:
        reader = QImageReader(path)
        # Ignore a wrong / missing extension and let Qt sniff the real format.
        reader.setDecideFormatFromContent(True)
        # Honour EXIF orientation (harmless for files without EXIF).
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            return QPixmap()
        return QPixmap.fromImage(image)
    except Exception:
        return QPixmap()
