# -*- coding: utf-8 -*-
"""
setup.py — ตั้งค่าเครื่องให้พร้อมใช้ ด้วยไฟล์เดียว

รันไฟล์นี้ไฟล์เดียว แล้วมันจะทำให้ครบทุกขั้น
    1. ตรวจเวอร์ชัน Python
    2. สร้าง Virtual Environment ที่โฟลเดอร์ .venv
    3. ติดตั้ง Library ทั้งหมด
    4. ตรวจว่าใช้งานได้จริง แล้วสรุปผลให้ดู

วิธีที่ง่ายที่สุด — เปิดไฟล์นี้ใน VS Code แล้วกดปุ่ม ▷ Run มุมขวาบน

หรือจะพิมพ์เองใน Terminal ที่โฟลเดอร์นี้ก็ได้
    python3 setup.py            # ครบทุกอย่าง — เหมือนกดปุ่ม Run
    python3 setup.py --core     # ข้ามชุด YOLOv5 ไปก่อน (ใช้ตอนเน็ตช้า)
    python3 setup.py --check    # ตรวจอย่างเดียว ไม่ติดตั้งอะไรใหม่

ไฟล์นี้ใช้เฉพาะ Library ที่ติดมากับ Python อยู่แล้ว จึงรันได้ทันทีก่อนติดตั้งอะไรทั้งสิ้น
"""
import os
import platform
import subprocess
import sys
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
IS_WIN = os.name == "nt"

OK, FAIL, WARN = "  [ ผ่าน ]", "  [ ไม่ผ่าน ]", "  [ ข้าม ]"


def line(char="─", n=64):
    print("  " + char * n)


def head(text):
    print("")
    line()
    print("  " + text)
    line()


def venv_python():
    """path ของ python ที่อยู่ข้างใน .venv"""
    return os.path.join(VENV, "Scripts" if IS_WIN else "bin",
                        "python.exe" if IS_WIN else "python")


def run(args, why):
    """เรียกคำสั่งแล้วคืน True/False — ถ้าพังจะพิมพ์เหตุผลให้อ่าน"""
    print("\n  → %s" % why)
    print("    $ %s" % " ".join(args))
    try:
        subprocess.check_call(args)
        return True
    except subprocess.CalledProcessError as e:
        print("%s %s (exit code %s)" % (FAIL, why, e.returncode))
        return False
    except FileNotFoundError:
        print("%s หาคำสั่งไม่เจอ: %s" % (FAIL, args[0]))
        return False


# ──────────────────────────────────────────────────────────── ขั้นตอน
def step_python():
    v = sys.version_info
    print("  Python ที่ใช้รันไฟล์นี้ : %d.%d.%d" % (v.major, v.minor, v.micro))
    print("  ระบบปฏิบัติการ         : %s %s" % (platform.system(), platform.machine()))

    if v.major != 3 or v.minor < 8:
        print("%s ต้องใช้ Python 3.8 ขึ้นไป" % FAIL)
        print("    ดาวน์โหลด Python 3.10 ได้จากลิงก์ในสไลด์")
        return False
    if v.minor < 10:
        print("%s Python %d.%d เก่ากว่าที่คอร์สแนะนำ (3.10 หรือ 3.11) — ใช้ต่อได้ "
              "แต่ถ้าติดปัญหาให้ลง 3.10 เพิ่ม" % (WARN, v.major, v.minor))
    elif v.minor >= 12:
        print("%s Python %d.%d ใหม่เกินไป — pypylon อาจยังไม่มีตัวติดตั้งสำหรับเวอร์ชันนี้"
              % (WARN, v.major, v.minor))
        print("    ถ้าติดตั้งไม่ผ่าน ให้ลง Python 3.10 เพิ่มแล้วรันไฟล์นี้ด้วย 3.10 แทน")
    else:
        print("%s เวอร์ชัน Python ใช้ได้" % OK)
    return True


def step_venv():
    py = venv_python()
    if os.path.exists(py):
        print("%s มี .venv อยู่แล้ว ใช้ตัวเดิมต่อ" % OK)
        return True
    print("\n  → สร้าง Virtual Environment ที่ .venv")
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    except Exception as e:
        print("%s สร้าง .venv ไม่สำเร็จ: %s" % (FAIL, e))
        return False
    if not os.path.exists(py):
        print("%s สร้าง .venv แล้วแต่หา python ข้างในไม่เจอ" % FAIL)
        return False
    print("%s สร้าง .venv เรียบร้อย" % OK)
    return True


def step_install(req_file, label):
    py = venv_python()
    path = os.path.join(ROOT, req_file)
    if not os.path.exists(path):
        print("%s ไม่พบไฟล์ %s" % (FAIL, req_file))
        return False
    run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        "อัปเดต pip ให้เป็นเวอร์ชันล่าสุด")
    return run([py, "-m", "pip", "install", "-r", path], "ติดตั้ง %s" % label)


PROBES = [
    ("numpy", "import numpy; print(numpy.__version__)", True),
    ("opencv-python", "import cv2; print(cv2.__version__)", True),
    ("pypylon", "from pypylon import pylon; print(pylon.__doc__ and 'ok' or 'ok')", False),
]


def step_verify(with_yolo=True):
    """ลอง import ของจริงข้างใน .venv — เห็นเวอร์ชันถึงจะนับว่าผ่าน"""
    py = venv_python()
    if not os.path.exists(py):
        print("%s ยังไม่มี .venv — กดปุ่ม Run บนไฟล์นี้ก่อน" % FAIL)
        return False

    probes = list(PROBES)
    if with_yolo:
        probes.append(("torch", "import torch; print(torch.__version__)", False))
        probes.append(("torchvision",
                       "import torchvision; print(torchvision.__version__)", False))

    all_required_ok = True
    for name, code, required in probes:
        try:
            out = subprocess.check_output([py, "-c", code],
                                          stderr=subprocess.STDOUT, text=True)
            print("%s %-16s %s" % (OK, name, out.strip().splitlines()[-1]))
        except subprocess.CalledProcessError:
            if required:
                all_required_ok = False
                print("%s %-16s import ไม่ผ่าน" % (FAIL, name))
            else:
                print("%s %-16s ยังไม่ได้ติดตั้ง" % (WARN, name))
                if name == "pypylon":
                    print("      ใช้ได้เฉพาะงานที่ต้องต่อกล้อง — งานภาพนิ่งยังทำได้ตามปกติ")
                if name in ("torch", "torchvision"):
                    print("      ใช้เฉพาะ Workshop #3 — Workshop #2 ยังทำได้ตามปกติ")
                    print("      ลองกด Run ซ้ำอีกครั้ง ถ้าเน็ตหลุดตอนโหลด")

    # เครื่องนี้จะรันโมเดลด้วยอะไร — บอกไว้ตั้งแต่ตอนติดตั้ง จะได้ไม่ต้องเดา
    if with_yolo:
        dev_code = (
            "import torch;"
            "d = 'cuda' if torch.cuda.is_available() else ("
            "'mps' if getattr(torch.backends,'mps',None) and"
            " torch.backends.mps.is_available() else 'cpu');"
            "print(d)")
        try:
            dev = subprocess.check_output([py, "-c", dev_code],
                                          stderr=subprocess.DEVNULL,
                                          text=True).strip()
            note = {"cuda": "การ์ดจอ NVIDIA — เร็วที่สุด",
                    "mps": "GPU ในตัวของ Mac — เร็วกว่า CPU 2–3 เท่า",
                    "cpu": "ไม่มี GPU — ยังทำงานได้ปกติ ราว 8–9 FPS"}.get(dev, "")
            print("%s ตัวประมวลผล      %-8s %s" % (OK, dev.upper(), note))
        except Exception:
            pass

    # กล้องต่ออยู่หรือยัง (ไม่ผ่านก็ไม่เป็นไร ยังทำงานกับภาพนิ่งได้)
    cam = ("from pypylon import pylon;"
           "d=pylon.TlFactory.GetInstance().EnumerateDevices();"
           "print(d[0].GetModelName() if d else 'NONE')")
    try:
        out = subprocess.check_output([py, "-c", cam], stderr=subprocess.DEVNULL,
                                      text=True).strip()
        if out == "NONE":
            print("%s กล้อง            ยังไม่เจอกล้อง (ปิด pylon Viewer แล้วเสียบสายใหม่)" % WARN)
        else:
            print("%s กล้อง            %s" % (OK, out))
    except Exception:
        print("%s กล้อง            ตรวจไม่ได้ เพราะยังไม่มี pypylon" % WARN)

    return all_required_ok


def finish(ok):
    head("สรุป")
    if ok:
        print("  พร้อมใช้งานแล้ว")
        print("")
        print("  ขั้นต่อไป — เปิดไฟล์ใน VS Code แล้วกดปุ่ม  ▷ Run  ไล่ทีละขั้น")
        print("")
        print("  Workshop #2  ที่โฟลเดอร์ Integration-Camera-Local/Pylon+Opencv")
        print("    step1_grab.py       เอาภาพเข้ามา            (ไม่ต้องมีกล้อง)")
        print("    step2_color.py      แปลง BGR เป็น HSV")
        print("    step3_extract.py    คัดสีทีละชนิด")
        print("    step4_count.py      นับและแยกทรง")
        print("    step5_count_all.py  ครบทั้ง 8 ชนิด")
        print("    step6_camera.py     ต่อกล้อง Basler นับแบบสด")
        print("")
        print("  Workshop #3  ที่โฟลเดอร์ Integration-Camera-Local/Pylon+Yolov5")
        print("    step1_model.py      ตรวจว่าโมเดลโหลดได้     (ทำก่อนเสมอ)")
        print("    step2_image.py      ภาพนิ่ง")
        print("    step3_video.py      วิดีโอ")
        print("    step4_camera.py     กล้อง Basler แบบ Real-time")
        print("")
        print("  ค่าที่ปรับได้ทั้งหมดของแต่ละ Workshop อยู่ในไฟล์ config.py ของโฟลเดอร์นั้น")
        print("")
        print("  ปุ่ม Run ใช้ .venv ให้เองอยู่แล้ว — ถ้าพิมพ์คำสั่งเอง ให้เปิด venv ก่อน")
        print("    %s" % ("  .venv\\Scripts\\Activate.ps1" if IS_WIN
                          else "  source .venv/bin/activate"))
    else:
        print("  ยังไม่พร้อม — ดูบรรทัดที่ขึ้น [ ไม่ผ่าน ] ด้านบน")
        print("  แก้แล้วกดปุ่ม Run ซ้ำได้เลย รันซ้ำกี่รอบก็ไม่เสียหาย")
    line()
    print("")


def main():
    args = sys.argv[1:]
    # กดปุ่ม Run เฉย ๆ = ไม่มี argument = ติดตั้งครบทุกอย่าง
    want_yolo = "--core" not in args
    check_only = "--check" in args

    head("ตั้งค่าเครื่องสำหรับคอร์ส Industrial Camera Integration & Deployment")

    if check_only:
        finish(step_verify(with_yolo=want_yolo))
        return

    if not step_python():
        finish(False)
        return
    if not step_venv():
        finish(False)
        return

    head("ติดตั้ง Library  (1 จาก %d)" % (2 if want_yolo else 1))
    ok = step_install("requirements.txt", "ชุดหลัก · กล้อง + OpenCV")

    if want_yolo:
        head("ติดตั้ง Library  (2 จาก 2)")
        print("  ชุดนี้ไฟล์ใหญ่ (PyTorch ~2 GB) ใช้เวลาราว 5–15 นาที")
        print("  ปล่อยให้มันโหลดไป อย่าเพิ่งปิดหน้าต่าง")
        # ถ้าชุด YOLOv5 ลงไม่สำเร็จ ยังถือว่าใช้งาน Workshop #2 ได้ จึงไม่ตัดสินว่า fail
        if not step_install("requirements-yolo.txt", "ชุด YOLOv5 · PyTorch"):
            print("%s ติดตั้งชุด YOLOv5 ไม่สำเร็จ — Workshop #2 ยังทำได้ตามปกติ" % WARN)
            print("    ลองกดปุ่ม Run ซ้ำอีกรอบ หรือเช็คว่าเน็ตหลุดระหว่างโหลดหรือเปล่า")

    head("ตรวจว่าใช้งานได้จริง")
    ok = step_verify(with_yolo=want_yolo) and ok
    finish(ok)


if __name__ == "__main__":
    main()
