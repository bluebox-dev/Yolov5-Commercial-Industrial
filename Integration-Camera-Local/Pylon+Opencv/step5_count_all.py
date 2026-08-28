# -*- coding: utf-8 -*-
"""
step5_count_all.py — Workshop #2 · ขั้นที่ 5  ·  ครบทั้ง 8 ชนิดในภาพเดียว

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  เอาสิ่งที่ทำมาตั้งแต่ขั้นที่ 1 ถึง 4 มาวนซ้ำให้ครบทั้ง 8 ชนิด    │
│  ได้ผลลัพธ์แบบเดียวกับที่ใช้งานจริง — กรอบ ชื่อชนิด และตารางสรุป  │
╰──────────────────────────────────────────────────────────────────╯

ขั้นนี้ไม่มีอะไรใหม่เลย มีแค่ลูป for วนทีละชนิด
    for ชื่อชนิด, ช่วงสี in COLORS:
        mask  = คัดสีนั้นออกมา        ← ขั้นที่ 3
        items = นับก้อนใน mask        ← ขั้นที่ 4

ที่ผลลัพธ์ดูซับซ้อนขึ้น เป็นเพราะข้อมูลเยอะขึ้น ไม่ใช่เพราะวิธีเปลี่ยน

วิธีรัน — กดปุ่ม ▷ Run มุมขวาบน

ปุ่มในหน้าต่างภาพ
    q   ปิดโปรแกรม
    s   เซฟภาพผลลัพธ์เป็น result.jpg
    m   กดวนดู mask ของแต่ละชนิดทีละอัน — ดูว่าช่วงสีที่ตั้งไว้จับอะไรได้บ้าง
    p   พิมพ์ตารางสรุปซ้ำ

ผลที่ควรได้จาก Candy.png คือ 21 เม็ด — ถ้าได้ไม่ตรง ให้ย้อนกลับไปขั้นที่ 3
"""
import cv2

import candy_lib as L
import config

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
#  (ค่าที่ปรับได้ทั้งหมดอยู่ใน config.py)
# ══════════════════════════════════════════════════════════════════


def main():
    bgr = L.load_image()
    print("\n  ภาพที่ใช้: %s" % L.here(config.IMAGE_PATH))
    print("  ตรวจทั้งหมด %d ชนิด" % len(config.COLORS))

    # ทั้งบรรทัดนี้คือลูป for ที่วนทำขั้นที่ 3 และ 4 ให้ครบทุกชนิด
    found = L.detect_all(bgr)

    L.print_table(found)
    view, total = L.annotate_all(bgr, found)

    names = list(config.COLORS)
    mask_at = -1                  # -1 = ภาพผลลัพธ์, 0 ขึ้นไป = mask ทีละชนิด
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    while True:
        if mask_at < 0:
            frame = view
        else:
            name = names[mask_at]
            mask = L.make_mask(hsv, config.COLORS[name])
            frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            L.banner(frame, "mask %d/%d: %s  (%d found)"
                     % (mask_at + 1, len(names), name,
                        len(L.find_items(mask))))

        cv2.imshow("step5 · count all", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            # กดวนไปเรื่อย ๆ ครบแล้วกลับมาที่ภาพผลลัพธ์
            mask_at = -1 if mask_at >= len(names) - 1 else mask_at + 1
        if key == ord("p"):
            L.print_table(found)
        if key == ord("s"):
            L.save("result.jpg", view)

    cv2.destroyAllWindows()
    print("  นับได้ทั้งหมด %d เม็ด" % total)
    print("\n  ขั้นที่ 5 จบแล้ว — มีกล้องแล้วไปต่อที่ step6_camera.py\n")


if __name__ == "__main__":
    main()
