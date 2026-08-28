# -*- coding: utf-8 -*-
"""
step2_color.py — Workshop #2 · ขั้นที่ 2  ·  CONVERT COLOR — จาก BGR ไป HSV

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  ให้เห็นว่าทำไมงานแยกสีต้องแปลงเป็น HSV ก่อน                     │
│  และได้ค่า H, S, V ของลูกอมจริงมาไว้ใช้ในขั้นที่ 3               │
╰──────────────────────────────────────────────────────────────────╯

BGR แยกสียาก เพราะ "สีอะไร" กับ "สว่างแค่ไหน" ปนกันอยู่ในทั้งสามช่อง
พอเงาตกลงบนชิ้นงาน ค่าทั้งสามช่องเปลี่ยนพร้อมกันหมด

HSV แยกสองเรื่องนี้ออกจากกัน
    H (Hue)        เฉดสี      0–179   ← เขียวก็คือเขียว ต่อให้อยู่ในเงา
    S (Saturation) ความสด     0–255
    V (Value)      ความสว่าง  0–255   ← เงาไปกระทบช่องนี้ช่องเดียว

วิธีรัน — กดปุ่ม ▷ Run มุมขวาบน

การใช้งาน
    คลิกซ้ายบนลูกอมเม็ดไหนก็ได้   อ่านค่า H S V ของจุดนั้นในหน้าจอ Terminal
    กด 1  ดูภาพต้นฉบับ BGR
    กด 2  ดูช่อง H (เฉดสี)
    กด 3  ดูช่อง S (ความสด)
    กด 4  ดูช่อง V (ความสว่าง)
    กด q  ปิดโปรแกรม
"""
import cv2

import candy_lib as L

# ══════════════════════════════════════════════════════════════════
#  ค่าที่ปรับได้ของขั้นนี้
# ══════════════════════════════════════════════════════════════════

SAMPLE_BOX = 5      # ตอนคลิก จะเฉลี่ยค่าจากสี่เหลี่ยมกว้างกี่พิกเซล
                    # เฉลี่ยหลายพิกเซลทำให้ค่านิ่งกว่าอ่านจุดเดียว

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
# ══════════════════════════════════════════════════════════════════

# ป้ายซ้ายคือข้อความที่ "วาดลงภาพ" ต้องเป็น ASCII ล้วน (cv2.putText วาดไทยไม่ได้)
# ป้ายขวาคือคำอธิบายไทย ที่จะพิมพ์ลงหน้าจอ Terminal แทน
VIEWS = {
    ord("1"): ("BGR original", "ภาพต้นฉบับ BGR", None),
    ord("2"): ("H  Hue",        "ช่อง H · เฉดสี", 0),
    ord("3"): ("S  Saturation", "ช่อง S · ความสด", 1),
    ord("4"): ("V  Value",      "ช่อง V · ความสว่าง", 2),
}


def on_click(event, x, y, flags, param):
    """อ่านค่า HSV ตรงจุดที่คลิก แล้วพิมพ์บรรทัดที่ก๊อปไปใช้ต่อได้เลย"""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    hsv = param
    h_img, w_img = hsv.shape[:2]
    r = SAMPLE_BOX // 2

    # กันไม่ให้กรอบตัวอย่างล้นออกนอกภาพตอนคลิกริมขอบ
    y1, y2 = max(0, y - r), min(h_img, y + r + 1)
    x1, x2 = max(0, x - r), min(w_img, x + r + 1)

    patch = hsv[y1:y2, x1:x2].reshape(-1, 3)
    h, s, v = patch.mean(axis=0)

    print("  คลิกที่ (%4d, %4d)   H=%3.0f   S=%3.0f   V=%3.0f"
          % (x, y, h, s, v))
    print("      ลองตั้งช่วงคร่อมค่านี้ใน config.py เช่น "
          "h=(%d, %d), s=(%d, 255), v=(%d, 255)"
          % (max(0, h - 10), min(179, h + 10), max(0, s - 60), max(0, v - 60)))


def main():
    bgr = L.load_image()

    # บรรทัดเดียวนี้คือทั้งหมดของการแปลง Color Space
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    print("\n  แปลงภาพเป็น HSV แล้ว — ขนาดเท่าเดิม %s" % (hsv.shape,))
    print("  ช่วงค่าที่วัดได้จากภาพนี้")
    for i, name in enumerate(("H (เฉดสี)  ", "S (ความสด) ", "V (ความสว่าง)")):
        ch = hsv[:, :, i]
        print("    %s  ต่ำสุด %3d   สูงสุด %3d   เฉลี่ย %3.0f"
              % (name, ch.min(), ch.max(), ch.mean()))

    print("\n  คลิกซ้ายบนลูกอมเพื่ออ่านค่าสีของเม็ดนั้น")
    print("  กด 1=BGR  2=H  3=S  4=V   ·   q=ปิด\n")

    win = "step2 · color space"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click, hsv)

    view = None                     # None = แสดงภาพ BGR ต้นฉบับ
    label = "BGR original"

    while True:
        if view is None:
            frame = bgr.copy()
        else:
            # ช่องเดียวเป็นภาพระดับเทา ต้องแปลงกลับเป็น 3 ช่องถึงจะเขียนตัวอักษรสีได้
            frame = cv2.cvtColor(hsv[:, :, view], cv2.COLOR_GRAY2BGR)

        L.banner(frame, "%s   (1=BGR  2=H  3=S  4=V  q=quit)" % label)
        cv2.imshow(win, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key in VIEWS:
            label, thai, view = VIEWS[key]
            print("  กำลังดู: %s" % thai)

    cv2.destroyAllWindows()
    print("\n  ขั้นที่ 2 จบแล้ว — ต่อไปกด Run ที่ step3_extract.py\n")


if __name__ == "__main__":
    main()
