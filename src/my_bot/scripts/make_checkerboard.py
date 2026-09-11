#!/usr/bin/env python3
"""Generate a checkerboard PDF at exact scale, for camera intrinsic calibration.

WHY THIS EXISTS. `--square` tells the calibrator how big a square is in metres.
That number is not measured, it is asserted -- and every intrinsic scales
linearly with it. A page printed at "Fit to page" comes out 3-6% small on A4,
and the calibration then reports an `fx` that is wrong by the same 3-6% while
looking perfectly healthy: the reprojection error stays low, because a uniform
scale error is absorbed by the board-to-camera distance. Nothing downstream
catches it either. `camera-lidar_semantic_mapping.md` converts a detection's
pixel extent to a bearing through `fx`, so a 5% `fx` error is a 5% bearing error
at the frame edge, which lands the semantic marker in the wrong place and looks
like a fusion bug for the rest of the week.

So: this writes a PDF with the squares in absolute PDF units (points, 1/72"),
which a printer honours as long as scaling is off, and it prints a 100 mm ruler
next to the board so you can prove that it did.

THE BOARD THIS PROJECT USES. 9x6 interior corners, 20 mm squares -- recovered
from the old board's `.bash_history` (see reference/nvme-recovery-audit.md).
"9x6" counts INTERIOR CORNERS, so the printed grid is 10x7 squares = 200x140 mm,
which is why the default page is A4 landscape. 9 is odd and 6 is even on
purpose: a chessboard with both dimensions odd or both even is rotationally
ambiguous and OpenCV can flip its corner ordering between frames.

USAGE

    ./make_checkerboard.py                       # 9x6 / 20 mm on A4 landscape
    ./make_checkerboard.py --out /tmp/board.pdf
    ./make_checkerboard.py --cols 9 --rows 6 --square 20 --page a4l

Then print it:

  * scaling MUST be "Actual size" / "100%" / "None" -- never "Fit to page"
  * check the printed ruler with a steel rule. 100 mm must be 100 mm.
  * if it is not, do not scale the PDF to compensate; fix the print dialog, or
    measure the squares you actually got and pass that to `--square` instead.

MOUNT IT ON SOMETHING RIGID AND FLAT. A sheet held in the hand bows by a
millimetre or two, which is 10% of a square, and it is the single most common
cause of a calibration that will not converge under 0.5 px. Glass, clipboard,
foamboard, a hardback book -- anything that cannot curl. Tape all four edges.

This script has no ROS and no OpenCV dependency; it writes the PDF bytes itself.
"""

import argparse
import sys

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0

# Page sizes in mm, (width, height).
PAGES = {
    "a4l": (297.0, 210.0),
    "a4": (210.0, 297.0),
    "letterl": (279.4, 215.9),
    "letter": (215.9, 279.4),
}


def mm2pt(mm):
    return mm * PT_PER_INCH / MM_PER_INCH


def content_stream(cols, rows, square_mm, page_w_mm, page_h_mm, top_margin_mm):
    """Draw the board, a 100 mm ruler and the caption. Returns a PDF content stream."""
    sq_x, sq_y = cols + 1, rows + 1          # printed squares, not corners
    board_w, board_h = sq_x * square_mm, sq_y * square_mm

    if board_w > page_w_mm or board_h > page_h_mm:
        sys.exit(
            "board is %.0fx%.0f mm and does not fit a %.0fx%.0f mm page.\n"
            "Use a bigger page (--page a4) or smaller squares -- do NOT let the\n"
            "printer scale it down, which is exactly the failure this avoids."
            % (board_w, board_h, page_w_mm, page_h_mm))

    x0 = (page_w_mm - board_w) / 2.0
    y0 = page_h_mm - top_margin_mm - board_h   # PDF origin is bottom-left

    ops = ["0 g"]                              # fill colour: black

    # The board. Square (i, j) is black when i+j is even, counting j from the
    # TOP row, which is the convention every checkerboard image uses. The
    # parity does not matter to OpenCV, only the alternation does.
    for j in range(sq_y):
        for i in range(sq_x):
            if (i + j) % 2:
                continue
            x = x0 + i * square_mm
            y = y0 + (sq_y - 1 - j) * square_mm
            ops.append("%.4f %.4f %.4f %.4f re f"
                       % (mm2pt(x), mm2pt(y), mm2pt(square_mm), mm2pt(square_mm)))

    # The ruler: a 100 mm baseline with a tick every 10 mm, taller at 0/50/100.
    # This is the only thing on the page that proves the print scale.
    ruler_len = 100.0
    rx = x0
    ry = y0 - 22.0
    ops.append("%.4f %.4f %.4f %.4f re f"
               % (mm2pt(rx), mm2pt(ry), mm2pt(ruler_len), mm2pt(0.6)))
    for k in range(11):
        tick_h = 5.0 if k % 5 == 0 else 3.0
        tx = rx + k * 10.0
        ops.append("%.4f %.4f %.4f %.4f re f"
                   % (mm2pt(tx), mm2pt(ry), mm2pt(0.6), mm2pt(tick_h)))

    def text(x_mm, y_mm, size, s):
        s = s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        return ("BT /F1 %.1f Tf %.4f %.4f Td (%s) Tj ET"
                % (size, mm2pt(x_mm), mm2pt(y_mm), s))

    ops.append(text(rx, ry - 6.0, 9.0,
                    "|<---- 100 mm ---->|   measure this with a steel rule "
                    "before you trust the calibration"))
    ops.append(text(rx, ry - 13.0, 11.0,
                    "%dx%d interior corners  -  %g mm squares  -  "
                    "%dx%d squares, %g x %g mm overall"
                    % (cols, rows, square_mm, sq_x, sq_y, board_w, board_h)))
    ops.append(text(rx, ry - 19.0, 9.0,
                    "print at 100%% / Actual size (NOT fit-to-page), then mount "
                    "flat and rigid.   cameracalibrator --size %dx%d --square %.3f"
                    % (cols, rows, square_mm / 1000.0)))

    return "\n".join(ops).encode("ascii")


def build_pdf(stream, page_w_mm, page_h_mm):
    """Assemble a minimal single-page PDF around an already-built content stream."""
    w, h = mm2pt(page_w_mm), mm2pt(page_h_mm)
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.4f %.4f] "
         "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
         % (w, h)).encode("ascii"),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, xref_at))
    return bytes(out)


def main():
    ap = argparse.ArgumentParser(
        description="Write an exact-scale checkerboard PDF for camera calibration.")
    ap.add_argument("--cols", type=int, default=9,
                    help="interior corners across (default 9 -- 10 squares)")
    ap.add_argument("--rows", type=int, default=6,
                    help="interior corners down (default 6 -- 7 squares)")
    ap.add_argument("--square", type=float, default=20.0,
                    help="square size in MILLIMETRES (default 20). Note that "
                         "cameracalibrator's --square is in metres.")
    ap.add_argument("--page", default="a4l", choices=sorted(PAGES),
                    help="page size (default a4l = A4 landscape)")
    ap.add_argument("--top-margin", type=float, default=14.0,
                    help="mm of white above the board (default 14). The quiet "
                         "zone matters: OpenCV needs white all round the grid.")
    ap.add_argument("--out", default="/tmp/checkerboard.pdf")
    args = ap.parse_args()

    if args.cols % 2 == args.rows % 2:
        print("WARNING: %dx%d has both dimensions the same parity. A chessboard "
              "like that is rotationally ambiguous and OpenCV can flip its corner "
              "order between frames. 9x6 is what this project uses."
              % (args.cols, args.rows), file=sys.stderr)

    page_w, page_h = PAGES[args.page]
    stream = content_stream(args.cols, args.rows, args.square,
                            page_w, page_h, args.top_margin)
    with open(args.out, "wb") as f:
        f.write(build_pdf(stream, page_w, page_h))

    print("wrote %s -- %dx%d interior corners, %g mm squares, %s"
          % (args.out, args.cols, args.rows, args.square, args.page))
    print()
    print("  1. print at 100%% / Actual size. NOT fit-to-page.")
    print("  2. measure the 100 mm ruler. If it is not 100 mm, reprint.")
    print("  3. mount it flat and rigid, tape all four edges.")
    print("  4. cameracalibrator --size %dx%d --square %.3f"
          % (args.cols, args.rows, args.square / 1000.0))


if __name__ == "__main__":
    main()
