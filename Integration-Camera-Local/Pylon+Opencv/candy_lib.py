# -*- coding: utf-8 -*-
"""
candy_lib.py — งานหลังบ้านที่ทุก step ใช้ร่วมกัน

ไฟล์นี้ไม่ต้องกด Run และปกติไม่ต้องแก้
มันเก็บเรื่องน่าเบื่อที่ซ้ำกันทุกไฟล์ไว้ที่เดียว — หาไฟล์ภาพ ย่อภาพ เปิดกล้อง วาดกรอบ
ส่วนตรรกะที่เป็นเนื้อหาของ Workshop จะเขียนไว้ในไฟล์ step ตรง ๆ ให้อ่านเห็นชัด

อ่านไฟล์นี้ตอนไหนดี — หลังจากทำ step1 ถึง step6 ครบแล้ว
"""
import os
import sys

import cv2
import numpy as np

import config

HERE = os.path.dirname(os.path.abspath(__file__))


def here(path):
    """แปลง path ให้นับจากโฟลเดอร์ของไฟล์นี้เสมอ

    จำเป็น เพราะปุ่ม Run ของ VS Code รันจากโฟลเดอร์นอกสุดของโปรเจกต์
    ไม่ใช่จากโฟลเดอร์ที่สคริปต์อยู่ — ถ้าใช้ path ตรง ๆ จะหาไฟล์ไม่เจอ
    """
    return path if os.path.isabs(path) else os.path.join(HERE, path)


# ─────────────────────────────────────────────────────────────────
#  อ่านภาพเข้ามา
# ─────────────────────────────────────────────────────────────────

def resize_work(img):
    """ย่อภาพให้กว้าง WORK_WIDTH เสมอ เพื่อให้ MIN_AREA มีความหมายคงที่"""
    h, w = img.shape[:2]
    if w == config.WORK_WIDTH:
        return img
    return cv2.resize(img, (config.WORK_WIDTH, int(h * config.WORK_WIDTH / w)))


def load_image(path=None):
    """อ่านภาพทดสอบแล้วย่อให้พร้อมใช้ — ถ้าหาไฟล์ไม่เจอจะบอกวิธีแก้แล้วหยุด"""
    full = here(path or config.IMAGE_PATH)

    if not os.path.exists(full):
        print("\n  หาไฟล์ภาพไม่เจอ: %s" % full)
        print("  วิธีแก้ — เลือกอย่างใดอย่างหนึ่ง")
        print("    1. ตรวจว่าดาวน์โหลดโปรเจกต์มาครบ (ต้องมี Integration-Camera-Local/Candy.png)")
        print("    2. แก้บรรทัด IMAGE_PATH ใน config.py ให้ชี้ไปที่ภาพของคุณ")
        print("    3. ถ่ายภาพเองด้วย step1_grab.py แล้วกด s เพื่อเซฟ\n")
        sys.exit(1)

    img = cv2.imread(full)
    if img is None:
        print("\n  เปิดไฟล์ไม่ได้: %s" % full)
        print("  ไฟล์อาจเสียหาย หรือไม่ใช่ไฟล์ภาพ\n")
        sys.exit(1)

    return resize_work(img)


# ─────────────────────────────────────────────────────────────────
#  ตรวจจับด้วยสี — step5 และ step6 เรียกใช้ชุดนี้
#  (ตรรกะเดียวกันนี้ step3 กับ step4 เขียนไว้แบบเต็ม ๆ ให้อ่านก่อนแล้ว)
# ─────────────────────────────────────────────────────────────────

def make_mask(hsv, spec):
    """สร้าง mask ขาว-ดำของชนิดหนึ่งชนิด — สีขาว = พิกเซลที่เข้าช่วงสีนี้"""
    h1, h2 = spec["h"]
    s1, s2 = spec["s"]
    v1, v2 = spec["v"]

    if h1 <= h2:
        mask = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))
    else:
        # ช่วงคร่อมเลข 0 (โซนสีแดง) ต้องตัดเป็นสองท่อนแล้วรวมกัน
        lo = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([179, s2, v2]))
        hi = cv2.inRange(hsv, np.array([0, s1, v1]), np.array([h2, s2, v2]))
        mask = cv2.bitwise_or(lo, hi)

    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.OPEN_SIZE, config.OPEN_SIZE))
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.CLOSE_SIZE, config.CLOSE_SIZE))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)     # ลบจุดรบกวน
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)   # อุดรู
    return mask


def find_items(mask):
    """หาชิ้นงานใน mask แล้วคืนตำแหน่ง ขนาด และทรง"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_AREA or area > config.MAX_AREA:
            continue

        (_, _), (rw, rh), _ = cv2.minAreaRect(c)      # กรอบที่หมุนตามชิ้นงาน
        ratio = max(rw, rh) / max(1.0, min(rw, rh))

        items.append({
            "contour": c,
            "area": area,
            "ratio": ratio,
            "shape": "LONG" if ratio >= config.LONG_RATIO else "ROUND",
            "rect": cv2.boundingRect(c),
        })
    return items


def detect_all(bgr):
    """ตรวจครบทุกชนิดในภาพเดียว คืนเป็น {ชื่อชนิด: [ชิ้นงาน...]}"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return {name: find_items(make_mask(hsv, spec))
            for name, spec in config.COLORS.items()}


# ─────────────────────────────────────────────────────────────────
#  แสดงผล
# ─────────────────────────────────────────────────────────────────

def annotate_all(bgr, found, hint="q=quit  s=save  m=mask"):
    """วาดกรอบและป้ายกำกับของทุกชนิดลงบนภาพ คืน (ภาพที่วาดแล้ว, จำนวนรวม)"""
    out = bgr.copy()
    total = 0

    for name, items in found.items():
        color = config.COLORS[name]["box"]
        for i, it in enumerate(items, 1):
            total += 1
            x, y, w, h = it["rect"]
            cv2.drawContours(out, [it["contour"]], -1, color, 2)
            cv2.putText(out, "%s %d" % (name, i), (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
            cv2.putText(out, it["shape"], (x, y + h + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    banner(out, "TOTAL %d   (%s)" % (total, hint))
    return out, total


def banner(img, text):
    """แถบดำบนหัวภาพ ใช้เขียนตัวเลขสรุปและปุ่มที่กดได้"""
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (25, 25, 25), -1)
    cv2.putText(img, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def print_table(found):
    """สรุปผลเป็นตารางในหน้าจอ Terminal คืนจำนวนรวม"""
    print("\n  ชนิด                              จำนวน   ทรงกลม  ทรงยาว")
    print("  " + "-" * 58)

    total = 0
    for name, items in found.items():
        rounds = sum(1 for i in items if i["shape"] == "ROUND")
        total += len(items)
        print("  %-18s %-12s %4d %7d %7d"
              % (name, config.COLORS[name]["th"], len(items), rounds,
                 len(items) - rounds))

    print("  " + "-" * 58)
    print("  รวมทั้งหมด %d เม็ด\n" % total)
    return total


def save(name, img):
    """เซฟภาพลงข้าง ๆ สคริปต์ แล้วบอก path ที่เซฟจริง"""
    path = here(name)
    cv2.imwrite(path, img)
    print("  เซฟแล้ว: %s" % path)


# ─────────────────────────────────────────────────────────────────
#  กล้อง Basler  (step1 โหมดกล้อง และ step6 เท่านั้นที่เรียกใช้)
# ─────────────────────────────────────────────────────────────────

def import_pylon():
    """import pypylon แบบรอจนถึงตอนใช้จริง

    ทำแบบนี้เพื่อให้ step ที่ใช้ภาพนิ่งรันได้แม้ยังไม่ได้ติดตั้ง pypylon
    """
    try:
        from pypylon import pylon
        return pylon
    except ImportError:
        print("\n  ยังไม่ได้ติดตั้ง pypylon — โหมดกล้องจึงยังใช้ไม่ได้")
        print("  วิธีแก้ — เปิด setup.py ที่โฟลเดอร์นอกสุด แล้วกดปุ่ม Run")
        print("\n  ยังไม่มีกล้อง? ใช้ step ที่เป็นภาพนิ่งไปก่อนได้ทั้งหมด\n")
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

    # กล้องส่งภาพมาเป็นรูปแบบของตัวเอง ต้องแปลงเป็น BGR ก่อน OpenCV ถึงใช้ได้
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return camera, converter


def set_manual(camera):
    """ปิด Auto แล้วตั้ง Exposure/Gain เอง — ค่าต้องนิ่ง ไม่งั้นตัวเลขจะแกว่ง"""
    for node, value in (("ExposureAuto", "Off"), ("GainAuto", "Off")):
        try:
            getattr(camera, node).SetValue(value)
        except Exception:
            pass                       # กล้องบางรุ่นไม่มี node นี้ ข้ามไปได้

    try:
        camera.ExposureTime.SetValue(config.EXPOSURE_US)
    except Exception:
        # กล้อง GigE รุ่นเก่าใช้ชื่อ ExposureTimeAbs ตาม GenICam เวอร์ชันก่อน
        try:
            camera.ExposureTimeAbs.SetValue(config.EXPOSURE_US)
        except Exception:
            print("  ตั้ง ExposureTime ไม่สำเร็จ — ใช้ค่าเดิมของกล้องแทน")

    try:
        camera.Gain.SetValue(config.GAIN_DB)
    except Exception:
        try:
            camera.GainRaw.SetValue(int(config.GAIN_DB))
        except Exception:
            pass


def grab_frames(camera, converter):
    """วนดึงเฟรมจากกล้องทีละใบ ย่อขนาดแล้วส่งออกเป็นภาพ BGR ของ OpenCV"""
    pylon = import_pylon()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    try:
        while camera.IsGrabbing():
            result = camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException)
            if not result.GrabSucceeded():
                result.Release()
                continue
            frame = resize_work(converter.Convert(result).GetArray())
            result.Release()
            yield frame
    finally:
        camera.StopGrabbing()
        camera.Close()
