# -*- coding: utf-8 -*-
"""
step1_grab.py — Workshop #2 · ขั้นที่ 1  ·  GRAB — เอาภาพเข้ามาให้ได้ก่อน

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  ให้เห็นว่า "เฟรมภาพ" ที่โปรแกรมได้รับ แท้จริงแล้วคือตารางตัวเลข  │
│  และเข้ามาได้ 2 ทาง — จากไฟล์ภาพ หรือจากกล้อง Basler             │
╰──────────────────────────────────────────────────────────────────╯

วิธีรัน — เปิดไฟล์นี้ใน VS Code แล้วกดปุ่ม ▷ Run มุมขวาบน

ปุ่มในหน้าต่างภาพ
    q   ปิดโปรแกรม
    s   เซฟภาพที่เห็นอยู่เป็น capture.jpg

ไม่มีกล้องก็ทำได้ครบ — ปล่อย SOURCE = "image" ไว้ตามเดิม
"""
import cv2

import candy_lib as L

# ══════════════════════════════════════════════════════════════════
#  ค่าที่ปรับได้ของขั้นนี้
# ══════════════════════════════════════════════════════════════════

SOURCE = "image"     # "image"  = อ่านจากไฟล์ Candy.png   (ไม่ต้องมีกล้อง)
                     # "camera" = ดึงภาพสดจากกล้อง Basler (ต้องมีกล้อง)

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
# ══════════════════════════════════════════════════════════════════


def describe(frame):
    """พิมพ์ให้เห็นว่าเฟรมหนึ่งใบมีอะไรอยู่ข้างในบ้าง"""
    h, w, ch = frame.shape
    print("\n  เฟรมนี้คือตารางตัวเลขขนาด %d x %d x %d" % (h, w, ch))
    print("    สูง       %d พิกเซล" % h)
    print("    กว้าง     %d พิกเซล" % w)
    print("    ช่องสี    %d ช่อง — OpenCV เรียงเป็น B, G, R ไม่ใช่ R, G, B" % ch)
    print("    ชนิดข้อมูล %s — แต่ละช่องเก็บค่า 0 ถึง 255" % frame.dtype)

    b, g, r = frame[h // 2, w // 2]      # หยิบพิกเซลกลางภาพมาดูสักจุด
    print("    พิกเซลกลางภาพ  B=%d  G=%d  R=%d" % (b, g, r))


def run_image():
    """ทางที่หนึ่ง — เอาภาพเข้ามาจากไฟล์"""
    print("\n  โหมดไฟล์ภาพ — อ่านจาก %s" % L.here(L.config.IMAGE_PATH))
    frame = L.load_image()
    describe(frame)

    print("\n  q = ปิด   s = เซฟภาพ")
    while True:
        view = L.banner(frame.copy(), "grab from file   (q=quit  s=save)")
        cv2.imshow("step1 · grab from file", view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            L.save("capture.jpg", frame)

    cv2.destroyAllWindows()


def run_camera():
    """ทางที่สอง — เอาภาพเข้ามาจากกล้อง Basler ทีละเฟรมแบบต่อเนื่อง"""
    print("\n  โหมดกล้อง — ต้องปิด pylon Viewer ก่อน กล้องเปิดได้ทีละโปรแกรม")
    camera, converter = L.open_camera()

    print("\n  q = ปิด   s = เซฟภาพ")
    first = True
    for frame in L.grab_frames(camera, converter):
        if first:                            # อธิบายแค่เฟรมแรกพอ
            describe(frame)
            first = False

        view = L.banner(frame.copy(), "grab from camera   (q=quit  s=save)")
        cv2.imshow("step1 · grab from camera", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            L.save("capture.jpg", frame)

    cv2.destroyAllWindows()


def main():
    if SOURCE == "camera":
        run_camera()
    else:
        run_image()

    print("\n  ขั้นที่ 1 จบแล้ว — ต่อไปกด Run ที่ step2_color.py\n")


if __name__ == "__main__":
    main()
