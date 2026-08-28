# -*- coding: utf-8 -*-
"""
step3_extract.py — Workshop #2 · ขั้นที่ 3  ·  COLOR EXTRACTION — คัดสีเดียวออกมา

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  คัดลูกอม "หนึ่งชนิด" ออกจากภาพให้ได้ก่อน แล้วทำความสะอาดผลลัพธ์  │
│  ขั้นนี้ยังไม่นับ — แค่ทำให้เหลือแต่สิ่งที่สนใจเป็นสีขาว          │
╰──────────────────────────────────────────────────────────────────╯

กระบวนการมี 2 ท่อน และไฟล์นี้เขียนไว้ให้เห็นเต็ม ๆ ทั้งสองท่อน

    1. cv2.inRange()      บอกว่าพิกเซลไหนอยู่ในช่วงสีที่กำหนด → ได้ภาพขาว-ดำ
    2. Morphology         ลบจุดรบกวน และอุดรูที่แสงสะท้อนเจาะไว้

วิธีรัน — กดปุ่ม ▷ Run มุมขวาบน

อยากเปลี่ยนชนิดที่ดู — แก้ ONE_CLASS ใน config.py แล้ว Run ใหม่

ปุ่มในหน้าต่างภาพ
    ลากแถบเลื่อน 6 ตัว   หาช่วงสีที่เหลือแต่ลูกอมชนิดนี้เป็นสีขาว
    p   พิมพ์บรรทัดค่าที่ได้ ก๊อปไปวางทับใน config.py ได้เลย
    r   กลับไปใช้ค่าเดิมที่อยู่ใน config.py
    q   ปิดโปรแกรม
"""
import cv2
import numpy as np

import candy_lib as L
import config

# ══════════════════════════════════════════════════════════════════
#  ค่าที่ปรับได้ของขั้นนี้
# ══════════════════════════════════════════════════════════════════

CLASS_NAME = config.ONE_CLASS    # ชนิดที่จะคัดออกมาดู — แก้ที่ config.py

SHOW_STAGES = True    # True = โชว์ 3 ภาพเรียงกัน (ต้นฉบับ · mask ดิบ · mask ที่ล้างแล้ว)
                      # False = โชว์เฉพาะผลสุดท้าย

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
# ══════════════════════════════════════════════════════════════════

BARS = ["H min", "H max", "S min", "S max", "V min", "V max"]
LIMITS = [179, 179, 255, 255, 255, 255]


# ── ท่อนที่ 1 · เลือกพิกเซลที่อยู่ในช่วงสี ────────────────────────
def color_mask(hsv, h, s, v):
    """คืนภาพขาว-ดำ — สีขาวคือพิกเซลที่อยู่ในช่วงสีที่กำหนด"""
    h1, h2 = h
    s1, s2 = s
    v1, v2 = v

    if h1 <= h2:
        # กรณีปกติ — ช่วงเดียวจบ
        return cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))

    # ช่วงคร่อมเลข 0 เช่นสีแดงที่พาดจาก 170 ไป 5
    # วงสีเป็นวงกลม แต่ inRange คิดเป็นเส้นตรง จึงต้องตัดสองท่อนแล้วรวมกัน
    lo = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([179, s2, v2]))
    hi = cv2.inRange(hsv, np.array([0, s1, v1]), np.array([h2, s2, v2]))
    return cv2.bitwise_or(lo, hi)


# ── ท่อนที่ 2 · ทำความสะอาด mask ──────────────────────────────────
def clean_mask(mask):
    """ลบจุดรบกวนแล้วอุดรู — ขั้นตอนนี้คือสิ่งที่ทำให้นับได้ตรงในขั้นที่ 4"""
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.OPEN_SIZE, config.OPEN_SIZE))
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.CLOSE_SIZE, config.CLOSE_SIZE))

    # OPEN  = กัดขอบเข้าแล้วขยายกลับ → จุดขาวเม็ดเล็กหายไป ชิ้นงานขนาดเท่าเดิม
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # CLOSE = ขยายก่อนแล้วกัดกลับ → รูดำกลางชิ้นงานถูกอุด ขอบนอกเท่าเดิม
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def stack(bgr, raw, clean):
    """เรียงภาพสามใบต่อกันในแนวนอน แล้วติดป้ายกำกับให้แต่ละใบ"""
    panels = [
        (bgr, "1 · BGR"),
        (cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), "2 · inRange"),
        (cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR), "3 · morphology"),
    ]

    out = []
    for img, text in panels:
        img = img.copy()
        cv2.putText(img, text, (12, img.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)
        out.append(img)

    # ย่อลงครึ่งหนึ่ง ไม่งั้นภาพสามใบต่อกันจะกว้างเกินจอ
    return cv2.resize(np.hstack(out), None, fx=0.5, fy=0.5)


def read_bars(win):
    """อ่านค่าแถบเลื่อนทั้ง 6 ตัว คืนเป็น (h, s, v) ที่เป็นคู่ค่าต่ำ-สูง"""
    p = [cv2.getTrackbarPos(b, win) for b in BARS]
    return (p[0], p[1]), (p[2], p[3]), (p[4], p[5])


def set_bars(win, spec):
    """ตั้งแถบเลื่อนกลับไปตามค่าใน config.py"""
    values = [spec["h"][0], spec["h"][1], spec["s"][0],
              spec["s"][1], spec["v"][0], spec["v"][1]]
    for bar, value in zip(BARS, values):
        cv2.setTrackbarPos(bar, win, value)


def main():
    if CLASS_NAME not in config.COLORS:
        print("\n  ไม่มีชนิดชื่อ '%s' ในตาราง COLORS" % CLASS_NAME)
        print("  แก้ ONE_CLASS ใน config.py ให้เป็นชื่อใดชื่อหนึ่งต่อไปนี้")
        print("    %s\n" % ", ".join(config.COLORS))
        raise SystemExit(1)

    spec = config.COLORS[CLASS_NAME]
    bgr = L.load_image()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    print("\n  กำลังคัดชนิด: %s (%s)" % (CLASS_NAME, spec["th"]))
    print("  ค่าตั้งต้นจาก config.py — h=%s  s=%s  v=%s"
          % (spec["h"], spec["s"], spec["v"]))
    print("\n  ลากแถบเลื่อนจนเหลือแต่ลูกอมชนิดนี้เป็นสีขาว")
    print("  p = พิมพ์ค่าที่ได้   r = กลับค่าเดิม   q = ปิด\n")

    win = "step3 · extract %s" % CLASS_NAME
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1100, 640)

    for bar, top in zip(BARS, LIMITS):
        cv2.createTrackbar(bar, win, 0, top, lambda _v: None)
    set_bars(win, spec)

    while True:
        h, s, v = read_bars(win)

        raw = color_mask(hsv, h, s, v)     # ท่อนที่ 1
        clean = clean_mask(raw)            # ท่อนที่ 2

        if SHOW_STAGES:
            frame = stack(bgr, raw, clean)
        else:
            frame = cv2.bitwise_and(bgr, bgr, mask=clean)

        # นับพิกเซลขาว ให้เห็นว่าการล้าง mask ตัดอะไรทิ้งไปบ้าง
        L.banner(frame, "%s   inRange %d px  ->  morphology %d px"
                 % (CLASS_NAME, int(raw.sum() // 255), int(clean.sum() // 255)))
        cv2.imshow(win, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            set_bars(win, spec)
            print("  กลับไปใช้ค่าเดิมจาก config.py แล้ว")
        if key == ord("p"):
            print('\n    "%s": dict(h=(%d, %d), s=(%d, %d), v=(%d, %d), '
                  'box=%s, th="%s"),'
                  % (CLASS_NAME, h[0], h[1], s[0], s[1], v[0], v[1],
                     spec["box"], spec["th"]))
            print("      ↑ ก๊อปบรรทัดนี้ไปวางทับบรรทัด %s ใน config.py\n"
                  % CLASS_NAME)

    cv2.destroyAllWindows()
    print("\n  ขั้นที่ 3 จบแล้ว — ต่อไปกด Run ที่ step4_count.py\n")


if __name__ == "__main__":
    main()
