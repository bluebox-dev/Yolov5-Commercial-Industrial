# -*- coding: utf-8 -*-
"""
step4_camera.py — Workshop #3 · ขั้นที่ 4  ·  กล้อง Basler + YOLOv5 แบบ Real-time

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  นี่คือปลายทางของทั้งคอร์ส — กล้องอุตสาหกรรมป้อนภาพเข้าโมเดล      │
│  ที่เราเทรนเอง แล้วตัดสินผลทันทีทีละเฟรม                          │
╰──────────────────────────────────────────────────────────────────╯

เทียบกับ step3_video.py ต่างกันแค่ที่มาของเฟรมอีกครั้ง

    step3 :  ok, frame = cap.read()                 อ่านจากไฟล์วิดีโอ
    step4 :  result = camera.RetrieveResult(...)    ดึงจากกล้องจริง

สิ่งที่เพิ่มมาในขั้นนี้คือการ "วัดว่าเวลาหมดไปกับอะไร"
    grab  = เวลาที่รอกล้องส่งภาพมา — สูงผิดปกติแปลว่าติดที่ Bandwidth
                                      หรือ Exposure Time ที่ตั้งไว้ยาวเกินไป
    infer = เวลาที่โมเดลคิด          — สูงเป็นเรื่องปกติ มักกินราว 90% ของทั้งหมด

    อย่าเพิ่งเร่งถ้ายังไม่ได้วัด — แก้จุดที่ไม่ใช่คอขวดคือเสียเวลาเปล่า

⚠ ต้องปิด pylon Viewer ก่อนรันทุกครั้ง — กล้องเปิดได้ทีละโปรแกรมเท่านั้น
⚠ ถ้าแสงและมุมกล้องต่างจากตอนเก็บ dataset ความแม่นจะตกลง เป็นเรื่องปกติ
   ทางแก้คือเก็บภาพจากหน้างานจริงเพิ่มแล้วเทรนใหม่ ไม่ใช่ไล่ลด conf

วิธีรัน — กดปุ่ม ▷ Run มุมขวาบน

ปุ่มในหน้าต่างภาพ
    q   ปิดโปรแกรม
    s   เซฟ 2 ไฟล์ — capture.jpg (ภาพดิบ) และ result.jpg (วาดกรอบแล้ว)
    p   พิมพ์ตารางสรุปลงหน้าจอ Terminal
"""
import time

import cv2

import yolo_lib as Y

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
#  (ค่าทั้งหมดอยู่ใน config.py)
# ══════════════════════════════════════════════════════════════════


def main():
    # โหลดโมเดลก่อนเปิดกล้องเสมอ
    # ถ้าเปิดกล้องก่อน กล้องจะถูกจองไว้เฉย ๆ ตลอดเวลาที่รอโมเดลโหลด
    model = Y.load_model()

    pylon = Y.import_pylon()
    camera, converter = Y.open_camera()

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("\n  q = ปิด   s = เซฟภาพ   p = พิมพ์ตารางสรุป\n")

    meter = Y.Meter()
    found = []
    warmed = False

    try:
        while camera.IsGrabbing():
            # ── ท่อนที่ 1 · Grab — ดึงเฟรมจากกล้อง
            t0 = time.time()
            result = camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException)
            if not result.GrabSucceeded():
                result.Release()
                continue
            frame = Y.resize_work(converter.Convert(result).GetArray())
            result.Release()

            # warm-up ด้วยเฟรมแรกจากกล้องจริง แล้วข้ามไปเฟรมถัดไปโดยไม่จับเวลา
            # ไม่งั้นตัวเลข grab กับ infer ของเฟรมแรกจะสูงผิดปกติจนอ่านผิด
            if not warmed:
                Y.warmup(model, frame)
                warmed = True
                meter.tick()          # รีเซ็ตนาฬิกา ไม่ให้รอบนี้ถูกนับ
                continue

            t1 = time.time()

            # ── ท่อนที่ 2 · Inference — ให้โมเดลตัดสิน
            found = Y.detect(model, frame)
            t2 = time.time()

            # ── ท่อนที่ 3 · แสดงผล
            view = Y.annotate(frame, found, "q=quit  s=save  p=table")

            meter.part("grab", t1 - t0)
            meter.part("infer", t2 - t1)
            meter.tick()
            cv2.putText(view, meter.text(), (view.shape[1] - 330, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

            cv2.imshow("step4 · yolo camera", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                Y.print_table(found)
            if key == ord("s"):
                Y.save("capture.jpg", frame)
                Y.save("result.jpg", view)
    finally:
        # ต้องปิดกล้องให้เรียบร้อยเสมอ ไม่งั้นรันรอบหน้าจะเปิดกล้องไม่ได้
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

    print("\n  จบ Workshop #3 — ครบทั้งคอร์สแล้ว\n")


if __name__ == "__main__":
    main()
