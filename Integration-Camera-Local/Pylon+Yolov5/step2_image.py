# -*- coding: utf-8 -*-
"""
step2_image.py — Workshop #3 · ขั้นที่ 2  ·  ตรวจภาพนิ่งด้วย YOLOv5

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  ใช้ภาพใบเดียวกับ Workshop #2 (Candy.png) แต่เปลี่ยนวิธีตรวจ      │
│  แล้วเอาผลสองวิธีมาวางเทียบกันตรง ๆ                              │
╰──────────────────────────────────────────────────────────────────╯

ต่างจาก Workshop #2 ตรงไหน
    Workshop #2  เราบอกเครื่องเองว่าสีเขียวคือ h อยู่ระหว่าง 55 ถึง 78
    Workshop #3  เราไม่ได้บอกค่าสีให้เลยสักตัว
                 โมเดลเรียนรู้เองทั้งหมดจากภาพที่เรา label ไว้ตอนเทรน

ผลที่ควรได้จาก Candy.png คือ 21 เม็ด เท่ากับวิธีแยกสีพอดี
แต่เม็ดที่ทับกันหรือวางเอียง YOLOv5 จะจัดการได้ดีกว่า

วิธีรัน — กดปุ่ม ▷ Run มุมขวาบน

อยากเปลี่ยนภาพ — แก้ IMAGE_SOURCE ใน config.py แล้ว Run ใหม่

ปุ่มในหน้าต่างภาพ
    q   ปิดโปรแกรม
    s   เซฟภาพผลลัพธ์เป็น result.jpg
    p   พิมพ์ตารางสรุปซ้ำ
"""
import time

import cv2

import yolo_lib as Y
import config

# ══════════════════════════════════════════════════════════════════
#  ค่าที่ปรับได้ของขั้นนี้
# ══════════════════════════════════════════════════════════════════

SOURCE = config.IMAGE_SOURCE     # แก้ที่ config.py หรือใส่ path ตรงนี้ก็ได้

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
# ══════════════════════════════════════════════════════════════════


def main():
    path = Y.need(SOURCE, "ไฟล์ภาพ",
                  "ตรวจบรรทัด IMAGE_SOURCE ใน config.py ว่าชี้ไปที่ไฟล์ที่มีอยู่จริง")

    model = Y.load_model()

    img = cv2.imread(path)
    if img is None:
        print("\n  เปิดไฟล์ภาพไม่ได้: %s" % path)
        print("  ไฟล์อาจเสียหาย หรือเป็นวิดีโอ — ถ้าเป็นวิดีโอให้ใช้ step3_video.py\n")
        raise SystemExit(1)

    img = Y.resize_work(img)
    print("\n  ภาพที่ใช้: %s   ขนาด %d x %d"
          % (path, img.shape[1], img.shape[0]))

    # วัดเวลาเฉพาะการตรวจ ไม่รวมเวลาอ่านไฟล์
    Y.warmup(model, img)          # warm-up ด้วยขนาดเดียวกับภาพจริง
    t0 = time.time()
    found = Y.detect(model, img)
    dt = (time.time() - t0) * 1000

    print("  ตรวจเสร็จใน %.0f ms  (%.1f FPS ถ้าทำต่อเนื่อง)" % (dt, 1000.0 / dt))
    Y.print_table(found)

    view = Y.annotate(img, found, "q=quit  s=save  p=table")

    while True:
        cv2.imshow("step2 · yolo image", view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            Y.print_table(found)
        if key == ord("s"):
            Y.save("result.jpg", view)

    cv2.destroyAllWindows()
    print("  ขั้นที่ 2 จบแล้ว — ต่อไปกด Run ที่ step3_video.py\n")


if __name__ == "__main__":
    main()
