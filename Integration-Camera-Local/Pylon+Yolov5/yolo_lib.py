# -*- coding: utf-8 -*-
"""
yolo_lib.py — งานหลังบ้านที่ทุก step ของ Workshop #3 ใช้ร่วมกัน

ไฟล์นี้ไม่ต้องกด Run และปกติไม่ต้องแก้
เก็บเรื่องที่ซ้ำกันทุกไฟล์ไว้ที่เดียว — โหลดโมเดล เลือกตัวประมวลผล วาดกรอบ เปิดกล้อง

⚠ สองบรรทัดแรกสุดของไฟล์นี้ต้องอยู่ "ก่อน" import torch เสมอ ห้ามสลับที่
   ถ้าย้ายลงไปข้างล่าง จะโหลด best.pt ไม่ได้และขึ้น error เรื่อง weights_only
"""
import os

# torch ตั้งแต่ 2.6 เปลี่ยนค่า default ของ torch.load เป็น weights_only=True
# ทำให้โหลดไฟล์ .pt ของ YOLOv5 ไม่ได้ ต้องปิดพฤติกรรมนี้ก่อน
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# YOLOv5 v7.0 จะพยายามสั่ง pip install เองเมื่อเจอ library ขาด
# ปิดไว้ เพราะ setup.py ลงให้ครบแล้ว และเครื่องในโรงงานมักไม่ได้ต่อเน็ต
os.environ["YOLOv5_AUTOINSTALL"] = "false"

import sys      # noqa: E402
import time     # noqa: E402

import cv2      # noqa: E402

import config   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def here(path):
    """แปลง path ให้นับจากโฟลเดอร์ของไฟล์นี้เสมอ

    จำเป็น เพราะปุ่ม Run ของ VS Code รันจากโฟลเดอร์นอกสุดของโปรเจกต์
    ไม่ใช่จากโฟลเดอร์ที่สคริปต์อยู่
    """
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def need(path, what, how):
    """เช็คว่าไฟล์ที่จำเป็นมีอยู่จริง ถ้าไม่มีก็บอกทางแก้แล้วหยุด"""
    full = here(path)
    if not os.path.exists(full):
        print("\n  หา%sไม่เจอ: %s" % (what, full))
        print("  %s\n" % how)
        sys.exit(1)
    return full


# ─────────────────────────────────────────────────────────────────
#  โหลดโมเดล
# ─────────────────────────────────────────────────────────────────

DEVICE_NOTE = {
    "cuda": "การ์ดจอ NVIDIA — เร็วที่สุด",
    "mps":  "GPU ในตัวของ Mac — เร็วกว่า CPU ราว 2–3 เท่า",
    "cpu":  "ไม่พบ GPU ที่ใช้ได้ — ยังทำงานได้ปกติ ราว 9 FPS",
}


def pick_device(torch):
    """เลือกตัวประมวลผลที่เร็วที่สุดเท่าที่เครื่องนี้มี"""
    if config.DEVICE != "auto":
        return config.DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(quiet=False):
    """โหลด YOLOv5 พร้อมน้ำหนัก best.pt แล้วย้ายไปยังตัวประมวลผลที่เร็วที่สุด

    ครั้งแรกต้องต่อเน็ตเพื่อดึงโค้ด YOLOv5 มาเก็บไว้ ครั้งต่อไปใช้ของที่เก็บไว้แล้ว
    """
    try:
        import torch
    except ImportError:
        print("\n  ยังไม่ได้ติดตั้ง PyTorch — ชุด YOLOv5 ยังไม่ได้ลง")
        print("  วิธีแก้ — เปิด setup.py ที่โฟลเดอร์นอกสุด แล้วกดปุ่ม Run")
        print("  ถ้าเคยกดแล้วยังไม่ได้ ให้กดซ้ำอีกรอบ มักเป็นเพราะเน็ตหลุดตอนโหลด\n")
        sys.exit(1)

    weights = need(config.MODEL_PATH, "ไฟล์โมเดล",
                   "เทรนจาก Colab-Training/YoloV5-Training.ipynb "
                   "แล้วเอา best.pt มาวางไว้ข้าง ๆ ไฟล์นี้")

    local = here("yolov5")     # ถ้า clone yolov5 มาวางเอง จะใช้ตัวนี้ก่อน
    print("  กำลังโหลดโมเดล... (ครั้งแรกอาจใช้เวลาสักครู่)")

    try:
        if os.path.isdir(local):
            model = torch.hub.load(local, "custom", path=weights, source="local")
        else:
            # ต้องปักที่ tag v7.0 ให้ตรงกับตอนเทรน
            # branch master ของ YOLOv5 เปลี่ยนไปเรียกแพ็กเกจ ultralytics แล้ว
            # ถ้าไม่ปัก tag จะโหลด best.pt ตัวนี้ไม่ได้
            model = torch.hub.load("ultralytics/yolov5:v7.0", "custom",
                                   path=weights, trust_repo=True)
    except Exception as e:
        print("\n  โหลดโมเดลไม่สำเร็จ: %s" % e)
        print("  ถ้าเครื่องไม่ได้ต่อเน็ต ให้ clone โค้ด YOLOv5 v7.0 มาวางไว้ก่อน")
        print("    git clone -b v7.0 https://github.com/ultralytics/yolov5")
        print("  วางไว้ในโฟลเดอร์เดียวกับไฟล์นี้ แล้วรันใหม่\n")
        sys.exit(1)

    model.conf = config.CONF
    model.iou = config.IOU

    dev = pick_device(torch)
    if dev != "cpu":
        try:
            model.to(dev)
        except Exception as e:
            print("  ใช้ %s ไม่สำเร็จ (%s) — ถอยไปใช้ cpu แทน"
                  % (dev, str(e)[:60]))
            dev = "cpu"
            model.to(dev)

    model.device_name = dev        # เก็บไว้ให้ step อื่นเอาไปแสดงผลได้

    if not quiet:
        print("  ตัวประมวลผล: %s  (%s)" % (dev.upper(), DEVICE_NOTE[dev]))
        print("  โหลดโมเดลเสร็จ — รู้จัก %d ชนิด" % len(model.names))
        for i, name in model.names.items():
            print("      %d · %s" % (i, name))

    return model


def warmup(model, sample=None, rounds=2):
    """รันโมเดลเปล่า ๆ สองสามรอบก่อนวัดเวลาจริง

    เฟรมแรกช้ากว่าปกติมากเพราะยังต้องจองหน่วยความจำและ compile kernel
    ถ้าไม่ warm-up ตัวเลข FPS ที่วัดได้ตอนต้นจะต่ำกว่าความจริงหลายเท่า

    ⚠ ต้องส่ง sample ที่ "ขนาดเท่ากับภาพจริง" ที่จะใช้เสมอ
       GPU จะ compile ใหม่ทุกครั้งที่ขนาดภาพเปลี่ยน — warm-up ด้วยขนาดอื่น
       จึงไม่ช่วยอะไรเลย และทำให้เฟรมแรกของจริงยังช้าอยู่ดี
    """
    import numpy as np

    if sample is None:
        sample = np.zeros((config.IMGSZ, config.IMGSZ, 3), dtype=np.uint8)
    else:
        # ใช้แค่ "ขนาด" ของภาพตัวอย่าง ไม่ต้องใช้เนื้อภาพจริง
        sample = np.zeros(sample.shape, dtype=np.uint8)

    for _ in range(rounds):
        model(sample, size=config.IMGSZ)


# ─────────────────────────────────────────────────────────────────
#  ตรวจจับและแสดงผล
# ─────────────────────────────────────────────────────────────────

def resize_work(img):
    """ย่อภาพให้กว้าง WORK_WIDTH เสมอ"""
    h, w = img.shape[:2]
    if w == config.WORK_WIDTH:
        return img
    return cv2.resize(img, (config.WORK_WIDTH, int(h * config.WORK_WIDTH / w)))


def detect(model, bgr):
    """ส่งภาพเข้าโมเดล แล้วคืนสิ่งที่เจอเป็น list ของ dict"""
    # ⚠ จุดที่พลาดกันมากที่สุด — OpenCV ให้ภาพมาเป็น BGR
    #    แต่โมเดลเทรนมาบน RGB ลืมแปลงตรงนี้จะไม่ error แต่ความแม่นตกฮวบ
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    results = model(rgb, size=config.IMGSZ)

    # results.xyxy[0] คือตาราง [x1, y1, x2, y2, confidence, class] ต่อหนึ่งวัตถุ
    found = []
    for *xyxy, conf, cls in results.xyxy[0].tolist():
        found.append({
            "name": model.names[int(cls)],
            "conf": float(conf),
            "box": [int(v) for v in xyxy],
        })
    return found


def annotate(bgr, found, hint="q=quit  s=save"):
    """วาดกรอบพร้อมชื่อชนิดและความมั่นใจ"""
    out = bgr.copy()

    for it in found:
        x1, y1, x2, y2 = it["box"]
        color = config.BOX.get(it["name"], (0, 200, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = "%s %.2f" % (it["name"], it["conf"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # พื้นทึบหลังตัวอักษร ไม่งั้นอ่านไม่ออกเมื่อกรอบทับกับลายชิ้นงาน
        cv2.rectangle(out, (x1, y1 - th - 7), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    banner(out, "TOTAL %d   (%s)" % (len(found), hint))
    return out


def banner(img, text):
    """แถบดำบนหัวภาพ ใช้เขียนตัวเลขสรุปและปุ่มที่กดได้"""
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (25, 25, 25), -1)
    cv2.putText(img, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def print_table(found):
    """สรุปจำนวนแยกตามชนิด พร้อมความมั่นใจต่ำสุด

    ตัวที่ conf ต่ำสุดคือตัวที่ควรเอาไปดูด้วยตา — มักเป็นชนิดที่ dataset มีน้อย
    """
    counts = {}
    for it in found:
        counts.setdefault(it["name"], []).append(it["conf"])

    print("\n  ชนิด                 จำนวน   conf ต่ำสุด  conf เฉลี่ย")
    print("  " + "-" * 52)
    for name in sorted(counts):
        cs = counts[name]
        print("  %-18s %5d %10.2f %11.2f"
              % (name, len(cs), min(cs), sum(cs) / len(cs)))
    print("  " + "-" * 52)
    print("  รวมทั้งหมด %d เม็ด\n" % len(found))


def save(name, img):
    """เซฟภาพลงข้าง ๆ สคริปต์ แล้วบอก path ที่เซฟจริง"""
    path = here(name)
    cv2.imwrite(path, img)
    print("  เซฟแล้ว: %s" % path)


class Meter:
    """นับ FPS และเวลาที่ใช้ในแต่ละท่อน แบบเฉลี่ยนุ่ม ๆ ไม่ให้ตัวเลขกระโดด"""

    def __init__(self):
        self.fps = 0.0
        self.parts = {}
        self._last = time.time()

    def part(self, name, seconds):
        ms = seconds * 1000
        self.parts[name] = 0.9 * self.parts.get(name, ms) + 0.1 * ms

    def tick(self):
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(1e-6, now - self._last)
        self._last = now
        return self.fps

    def text(self):
        parts = "   ".join("%s %.0f ms" % (k, v) for k, v in self.parts.items())
        return "%.1f FPS   %s" % (self.fps, parts)


# ─────────────────────────────────────────────────────────────────
#  กล้อง Basler  (step4 เท่านั้นที่เรียกใช้)
# ─────────────────────────────────────────────────────────────────

def import_pylon():
    """import pypylon แบบรอจนถึงตอนใช้จริง

    ทำแบบนี้เพื่อให้ step1 ถึง step3 รันได้แม้ยังไม่ได้ติดตั้ง pypylon
    """
    try:
        from pypylon import pylon
        return pylon
    except ImportError:
        print("\n  ยังไม่ได้ติดตั้ง pypylon — โหมดกล้องจึงยังใช้ไม่ได้")
        print("  วิธีแก้ — เปิด setup.py ที่โฟลเดอร์นอกสุด แล้วกดปุ่ม Run")
        print("\n  ยังไม่มีกล้อง? ใช้ step2_image.py และ step3_video.py "
              "ได้ครบทุกอย่าง\n")
        sys.exit(1)


def open_camera():
    """เปิดกล้องตัวแรกที่เจอ โหลดค่าจาก .pfs แล้วคืน (camera, converter)"""
    pylon = import_pylon()

    factory = pylon.TlFactory.GetInstance()
    if not factory.EnumerateDevices():
        print("\n  ไม่พบกล้อง — ไล่เช็คตามนี้")
        print("    · ปิด pylon Viewer หรือยัง (กล้องเปิดได้ทีละโปรแกรมเท่านั้น)")
        print("    · กล้อง USB3 ต้องเสียบพอร์ต SS สีฟ้าโดยตรง ห้ามผ่าน USB hub")
        print("    · กล้อง GigE ต้องอยู่ subnet เดียวกับการ์ดแลน")
        print("    · ติดตั้ง pylon Runtime ครบหรือยัง\n")
        sys.exit(1)

    camera = pylon.InstantCamera(factory.CreateFirstDevice())
    camera.Open()
    print("  เชื่อมต่อกับ: %s" % camera.GetDeviceInfo().GetModelName())

    pfs = here(config.PFS_FILE)
    if os.path.exists(pfs):
        pylon.FeaturePersistence.Load(pfs, camera.GetNodeMap(), True)
        print("  โหลดค่ากล้องจาก %s แล้ว" % config.PFS_FILE)
    else:
        print("  ไม่เจอ %s — ใช้ค่าใน config.py แทน" % config.PFS_FILE)
        set_manual(camera)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return camera, converter


def set_manual(camera):
    """ปิด Auto แล้วตั้ง Exposure/Gain เอง — ค่าต้องนิ่ง ไม่งั้นความแม่นจะแกว่ง"""
    for node, value in (("ExposureAuto", "Off"), ("GainAuto", "Off")):
        try:
            getattr(camera, node).SetValue(value)
        except Exception:
            pass                       # กล้องบางรุ่นไม่มี node นี้ ข้ามไปได้

    try:
        camera.ExposureTime.SetValue(config.EXPOSURE_US)
    except Exception:
        try:
            camera.ExposureTimeAbs.SetValue(config.EXPOSURE_US)  # GigE รุ่นเก่า
        except Exception:
            print("  ตั้ง ExposureTime ไม่สำเร็จ — ใช้ค่าเดิมของกล้องแทน")

    try:
        camera.Gain.SetValue(config.GAIN_DB)
    except Exception:
        try:
            camera.GainRaw.SetValue(int(config.GAIN_DB))
        except Exception:
            pass
