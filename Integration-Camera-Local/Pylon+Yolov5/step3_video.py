# -*- coding: utf-8 -*-
"""
step3_video.py — Workshop #3 · ขั้นที่ 3  ·  ตรวจวิดีโอด้วย YOLOv5

╭──────────────────────────────────────────────────────────────────╮
│  เป้าหมายของขั้นนี้                                              │
│  เปลี่ยนจาก "ภาพหนึ่งใบ" เป็น "ภาพต่อเนื่อง" โดยไม่แก้ตรรกะการตรวจ │
│  และเห็นตัวเลข FPS จริงของเครื่องตัวเอง                          │
╰──────────────────────────────────────────────────────────────────╯

เทียบกับ step2_image.py จะเห็นว่าต่างกันแค่วิธีหาเฟรม

    step2 :  img = cv2.imread(path)              อ่านครั้งเดียวจบ
    step3 :  ok, frame = cap.read()              วนอ่านทีละเฟรมในลูป

ส่วน Y.detect() และ Y.annotate() ใช้ตัวเดียวกับ step2 เป๊ะ ๆ
เปลี่ยนแค่ "แหล่งที่มาของภาพ" — ตรรกะไม่ต้องแตะเลย

วิธีรัน — กดปุ่ม ▷ Run มุมขวาบน

อยากเปลี่ยนวิดีโอ — แก้ VIDEO_SOURCE ใน config.py แล้ว Run ใหม่
    มีให้เลือก: 1.mp4 · 2.mp4 · 3.mp4 · homework.mp4

ปุ่มในหน้าต่างภาพ
    q          ปิดโปรแกรม
    เว้นวรรค    หยุด / เล่นต่อ
    s          เซฟภาพเฟรมที่เห็นอยู่
    p          พิมพ์ตารางสรุปของเฟรมที่เห็นอยู่

รู้สึกว่าไม่ลื่น — ตั้ง FRAME_STRIDE = 2 ใน config.py จะข้ามเฟรมเว้นเฟรม
"""
import time

import cv2

import yolo_lib as Y
import config

# ══════════════════════════════════════════════════════════════════
#  ค่าที่ปรับได้ของขั้นนี้
# ══════════════════════════════════════════════════════════════════

SOURCE = config.VIDEO_SOURCE     # แก้ที่ config.py หรือใส่ path ตรงนี้ก็ได้

# ══════════════════════════════════════════════════════════════════
#  ตั้งแต่บรรทัดนี้ลงไปคือส่วนทำงาน
# ══════════════════════════════════════════════════════════════════


def main():
    path = Y.need(SOURCE, "ไฟล์วิดีโอ",
                  "ตรวจบรรทัด VIDEO_SOURCE ใน config.py ว่าชี้ไปที่ไฟล์ที่มีอยู่จริง")

    model = Y.load_model()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("\n  เปิดไฟล์วิดีโอไม่ได้: %s" % path)
        print("  ตรวจว่าไฟล์ไม่เสีย และเป็นสกุลที่ OpenCV อ่านได้ (.mp4 ทั่วไปใช้ได้)\n")
        raise SystemExit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print("\n  วิดีโอ: %s" % path)
    print("  %d เฟรม · %.0f fps ต้นฉบับ · ประมวลผลทุก %d เฟรม"
          % (total, src_fps, config.FRAME_STRIDE))
    print("\n  q = ปิด   เว้นวรรค = หยุด/เล่นต่อ   s = เซฟภาพ   p = ตารางสรุป\n")

    meter = Y.Meter()
    writer = None
    idx, paused, view, found = 0, False, None, []
    warmed = False

    while True:
        if not paused:
            ok, frame = cap.read()

            if not ok:                       # อ่านจนหมดไฟล์แล้ว
                if config.LOOP_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    idx = 0
                    continue
                break

            idx += 1
            if config.FRAME_STRIDE > 1 and idx % config.FRAME_STRIDE:
                continue                     # ข้ามเฟรมนี้ ไม่ต้องตรวจ

            frame = Y.resize_work(frame)

            # warm-up ด้วยเฟรมแรกของจริง แล้วข้ามไปเฟรมถัดไปโดยไม่จับเวลา
            # เพื่อไม่ให้เวลาของรอบ warm-up ไปปนกับตัวเลข FPS ที่แสดง
            if not warmed:
                Y.warmup(model, frame)
                warmed = True
                meter.tick()          # รีเซ็ตนาฬิกา ไม่ให้รอบนี้ถูกนับ
                continue

            t0 = time.time()
            found = Y.detect(model, frame)
            meter.part("infer", time.time() - t0)

            view = Y.annotate(frame, found, "q=quit  space=pause  s=save")

            meter.tick()
            cv2.putText(view, "%s   frame %d/%d" % (meter.text(), idx, total),
                        (view.shape[1] - 330, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if config.SAVE_VIDEO:
                if writer is None:
                    h, w = view.shape[:2]
                    writer = cv2.VideoWriter(
                        Y.here("result.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        src_fps / max(1, config.FRAME_STRIDE), (w, h))
                writer.write(view)

        if view is not None:
            cv2.imshow("step3 · yolo video", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            print("  %s" % ("หยุดชั่วคราว — กดเว้นวรรคอีกครั้งเพื่อเล่นต่อ"
                            if paused else "เล่นต่อ"))
        if key == ord("p"):
            Y.print_table(found)
        if key == ord("s") and view is not None:
            Y.save("result.jpg", view)

    cap.release()
    if writer is not None:
        writer.release()
        print("  เซฟวิดีโอผลลัพธ์แล้วที่ %s" % Y.here("result.mp4"))

    cv2.destroyAllWindows()
    print("\n  ขั้นที่ 3 จบแล้ว — มีกล้องแล้วไปต่อที่ step4_camera.py\n")


if __name__ == "__main__":
    main()
