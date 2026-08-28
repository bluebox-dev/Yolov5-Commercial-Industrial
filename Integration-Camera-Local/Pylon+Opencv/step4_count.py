# -*- coding: utf-8 -*-
"""
step4_count.py — Workshop #2 · ขั้นที่ 4  ·  COUNTING — นับและแยกทรง

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  เปลี่ยนภาพขาว-ดำจากขั้นที่ 3 ให้กลายเป็น "ตัวเลข"               │
│  ตอบให้ได้ว่า มีกี่เม็ด · แต่ละเม็ดอยู่ตรงไหน · ทรงกลมหรือทรงยาว  │
╰──────────────────────────────────────────────────────────────────╯

กระบวนการมี 3 ท่อน เขียนไว้ให้เห็นเต็ม ๆ ในไฟล์นี้

    1. cv2.findContours()   หาเส้นขอบของก้อนสีขาวทุกก้อน
    2. กรองด้วยพื้นที่        ก้อนที่เล็กหรือใหญ่เกินไปคือ noise ไม่ใช่ชิ้นงาน
    3. cv2.minAreaRect()    วัดสัดส่วนด้านยาว ÷ ด้านสั้น → ตัดสินว่าทรงอะไร

ทำไมต้องใช้ minAreaRect ไม่ใช่ boundingRect
    boundingRect ให้กรอบที่ตั้งตรงเสมอ ลูกอมทรงยาวที่วางเอียง 45 องศา
    จะได้กรอบเกือบเป็นสี่เหลี่ยมจัตุรัส แล้วถูกตัดสินผิดว่าเป็นทรงกลม
    ส่วน minAreaRect ให้กรอบที่หมุนตามชิ้นงาน จึงวัดสัดส่วนได้ตรงทุกมุมวาง

วิธีรัน — กดปุ่ม ▷ Run มุมขวาบน

ปุ่มในหน้าต่างภาพ
    q   ปิดโปรแกรม
    s   เซฟภาพผลลัพธ์
    m   สลับดู mask ที่ใช้นับ
"""
import cv2

import candy_lib as L
import config

# ══════════════════════════════════════════════════════════════════
#  ค่าที่ปรับได้ของขั้นนี้
# ══════════════════════════════════════════════════════════════════

CLASS_NAME = config.ONE_CLASS    # ชนิดที่จะนับ — แก้ที่ config.py

SHOW_AREA = True     # True = เขียนพื้นที่กับสัดส่วนของทุกเม็ดลงบนภาพ
                     #        เอาไว้ดูว่าควรตั้ง MIN_AREA เท่าไหร่

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
# ══════════════════════════════════════════════════════════════════


def count_items(mask):
    """หาชิ้นงานทุกชิ้นใน mask — หัวใจของการนับอยู่ที่ฟังก์ชันนี้"""

    # ท่อนที่ 1 · หาเส้นขอบของก้อนสีขาวทุกก้อน
    #   RETR_EXTERNAL      เอาเฉพาะขอบนอกสุด ไม่เอารูข้างใน
    #   CHAIN_APPROX_SIMPLE เก็บเฉพาะจุดหักมุม ประหยัดหน่วยความจำ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    print("  เจอก้อนสีขาวทั้งหมด %d ก้อน (ยังไม่กรอง)" % len(contours))

    items, dropped = [], 0
    for c in contours:
        area = cv2.contourArea(c)

        # ท่อนที่ 2 · กรองด้วยพื้นที่
        if area < config.MIN_AREA or area > config.MAX_AREA:
            dropped += 1
            continue

        # ท่อนที่ 3 · วัดสัดส่วนจากกรอบที่หมุนตามชิ้นงาน
        (_, _), (rw, rh), angle = cv2.minAreaRect(c)
        ratio = max(rw, rh) / max(1.0, min(rw, rh))

        items.append({
            "contour": c,
            "area": area,
            "ratio": ratio,
            "angle": angle,
            "shape": "LONG" if ratio >= config.LONG_RATIO else "ROUND",
            "rect": cv2.boundingRect(c),
        })

    print("  ตัดทิ้งเพราะขนาดไม่เข้าเกณฑ์ %d ก้อน" % dropped)
    print("  เหลือเป็นชิ้นงานจริง %d เม็ด" % len(items))
    return items


def draw(bgr, items, color):
    """วาดกรอบ ลำดับ และผลตัดสินทรง ลงบนภาพ"""
    out = bgr.copy()

    for i, it in enumerate(items, 1):
        x, y, w, h = it["rect"]
        cv2.drawContours(out, [it["contour"]], -1, color, 2)
        cv2.putText(out, "#%d %s" % (i, it["shape"]), (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if SHOW_AREA:
            cv2.putText(out, "area %d  ratio %.2f" % (it["area"], it["ratio"]),
                        (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        color, 1, cv2.LINE_AA)

    return out


def report(items):
    """พิมพ์ผลของทุกเม็ด ให้เห็นตัวเลขที่ใช้ตัดสิน"""
    rounds = sum(1 for i in items if i["shape"] == "ROUND")

    print("\n  เม็ดที่   พื้นที่ (px²)   สัดส่วน   ทรง")
    print("  " + "-" * 44)
    for i, it in enumerate(items, 1):
        print("  %5d %12d %9.2f   %s"
              % (i, it["area"], it["ratio"], it["shape"]))
    print("  " + "-" * 44)
    print("  รวม %d เม็ด — ทรงกลม %d · ทรงยาว %d"
          % (len(items), rounds, len(items) - rounds))
    print("  เกณฑ์ที่ใช้ตัดสิน: สัดส่วนตั้งแต่ %.1f ขึ้นไปคือทรงยาว\n"
          % config.LONG_RATIO)


def main():
    if CLASS_NAME not in config.COLORS:
        print("\n  ไม่มีชนิดชื่อ '%s' ในตาราง COLORS" % CLASS_NAME)
        print("  แก้ ONE_CLASS ใน config.py ให้เป็นชื่อใดชื่อหนึ่งต่อไปนี้")
        print("    %s\n" % ", ".join(config.COLORS))
        raise SystemExit(1)

    spec = config.COLORS[CLASS_NAME]
    bgr = L.load_image()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # ใช้ mask จากขั้นที่ 3 ตรง ๆ — ขั้นนี้สนใจแค่การนับ
    mask = L.make_mask(hsv, spec)

    print("\n  กำลังนับชนิด: %s (%s)" % (CLASS_NAME, spec["th"]))
    print("  เกณฑ์ขนาด: %d ถึง %d พิกเซล²\n"
          % (config.MIN_AREA, config.MAX_AREA))

    items = count_items(mask)
    report(items)

    view = draw(bgr, items, spec["box"])
    # ข้อความที่วาดลงภาพต้องเป็น ASCII ล้วน — cv2.putText วาดภาษาไทยไม่ได้
    L.banner(view, "%s  -  %d items   (q=quit  s=save  m=mask)"
             % (CLASS_NAME, len(items)))

    show_mask = False
    while True:
        if show_mask:
            frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            L.banner(frame, "mask used for counting   (m=back to result)")
        else:
            frame = view

        cv2.imshow("step4 · count", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask = not show_mask
        if key == ord("s"):
            L.save("result.jpg", view)

    cv2.destroyAllWindows()
    print("\n  ขั้นที่ 4 จบแล้ว — ต่อไปกด Run ที่ step5_count_all.py\n")


if __name__ == "__main__":
    main()
