# -*- coding: utf-8 -*-
"""
step6_camera.py — Workshop #2 · ขั้นที่ 6  ·  ต่อกล้อง Basler นับแบบสด

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  เปลี่ยนจาก "ภาพนิ่งหนึ่งใบ" เป็น "ภาพสดจากกล้อง" โดยไม่แก้ตรรกะ  │
│  ตรงไหนเลย — เปลี่ยนแค่ที่มาของเฟรมเท่านั้น                      │
╰──────────────────────────────────────────────────────────────────╯

เทียบกับ step5_count_all.py จะเห็นว่าต่างกันแค่สองบรรทัด

    step5 :  bgr = L.load_image()                  อ่านครั้งเดียวจากไฟล์
    step6 :  for bgr in L.grab_frames(...)         วนรับทีละเฟรมจากกล้อง

ส่วน detect_all() และ annotate_all() ใช้ตัวเดียวกันเป๊ะ
นี่คือเหตุผลที่ต้องแยกตรรกะออกจากที่มาของภาพตั้งแต่แรก

⚠ ต้องปิด pylon Viewer ก่อนรันทุกครั้ง — กล้องเปิดได้ทีละโปรแกรมเท่านั้น
⚠ ค่าสีที่ตั้งไว้วัดจากไฟล์ Candy.png ไฟในห้องคุณต่างออกไปแน่นอน
   ถ้านับไม่ตรง ให้กด s เซฟภาพดิบไว้ แล้วย้อนไปจูนที่ step3_extract.py

วิธีรัน — กดปุ่ม ▷ Run มุมขวาบน

ปุ่มในหน้าต่างภาพ
    q   ปิดโปรแกรม
    s   เซฟ 2 ไฟล์ — capture.jpg (ภาพดิบ) และ result.jpg (วาดกรอบแล้ว)
    m   กดวนดู mask ของแต่ละชนิดทีละอัน
    p   พิมพ์ตารางสรุปลงหน้าจอ Terminal
"""
import time

import cv2

import candy_lib as L
import config

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
#  (ค่ากล้อง PFS_FILE / EXPOSURE_US / GAIN_DB อยู่ใน config.py)
# ══════════════════════════════════════════════════════════════════


def main():
    camera, converter = L.open_camera()

    print("\n  q = ปิด   s = เซฟภาพ   m = ดู mask   p = พิมพ์ตารางสรุป\n")

    names = list(config.COLORS)
    mask_at = -1               # -1 = ภาพผลลัพธ์, 0 ขึ้นไป = mask ทีละชนิด
    fps, last = 0.0, time.time()
    found = {}

    for frame in L.grab_frames(camera, converter):
        # ตรรกะชุดเดียวกับ step5 ทุกบรรทัด
        found = L.detect_all(frame)
        view, total = L.annotate_all(frame, found)

        # เฉลี่ยแบบนุ่ม ๆ ไม่ให้ตัวเลข FPS กระโดดจนอ่านไม่ทัน
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-6, now - last)
        last = now
        cv2.putText(view, "%.1f FPS" % fps, (view.shape[1] - 110, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)

        if mask_at < 0:
            shown = view
        else:
            name = names[mask_at]
            mask = L.make_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                               config.COLORS[name])
            shown = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            L.banner(shown, "mask %d/%d: %s  (%d found)"
                     % (mask_at + 1, len(names), name,
                        len(L.find_items(mask))))

        cv2.imshow("step6 · camera live", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mask_at = -1 if mask_at >= len(names) - 1 else mask_at + 1
        if key == ord("p"):
            L.print_table(found)
        if key == ord("s"):
            L.save("capture.jpg", frame)
            L.save("result.jpg", view)
            print("  เอา capture.jpg ไปจูนค่าสีต่อได้ที่ step3_extract.py")
            print("  โดยแก้ IMAGE_PATH ใน config.py ให้ชี้มาที่ไฟล์นี้")

    cv2.destroyAllWindows()
    print("\n  จบ Workshop #2 — ต่อไปเป็น Workshop #3 ที่โฟลเดอร์ Pylon+Yolov5\n")


if __name__ == "__main__":
    main()
