# Third-Party Notices

เอกสารนี้ระบุใบอนุญาตของส่วนประกอบภายนอกทั้งหมดที่โปรเจกต์นี้ใช้หรือแจกจ่ายร่วมด้วย
ผู้ที่นำโค้ดไปพัฒนาต่อหรือแจกจ่าย **ต้องอ่านและปฏิบัติตามเงื่อนไขในตารางเหล่านี้**

โปรเจกต์นี้เอง (`Yolov5-Commercial-Industrial`) เผยแพร่ภายใต้ **GPL-3.0** — ดู [`LICENSE`](LICENSE)

```
Copyright (C) 2026 bluebox-dev  <https://github.com/bluebox-dev>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
```

---

## 1 · เหตุผลที่โปรเจกต์นี้เป็น GPL-3.0

ไม่ได้เลือกเอง แต่ถูกกำหนดโดยส่วนประกอบที่ใช้

| ข้อเท็จจริง | ผล |
|---|---|
| โค้ดเทรนและ inference อ้างอิง **YOLOv5 tag `v7.0`** ซึ่งเป็น GPL-3.0 | โค้ดที่ทำงานร่วมกันเป็น combined work → ต้อง GPL-3.0 |
| `Integration-Camera-Local/Pylon+Yolov5/best.pt` ถูกแจกจ่ายมาใน repo นี้ | weights เทรนด้วย YOLOv5 v7.0 และ checkpoint อ้างคลาสของ YOLOv5 ตอนโหลด → ถือเป็น GPL-3.0 ตามต้นทาง |
| YOLOv5 ตั้งแต่ 14 เม.ย. 2023 เปลี่ยนเป็น **AGPL-3.0** | โปรเจกต์นี้ **ปักที่ v7.0 ทุกจุด** จึงไม่ตกอยู่ใต้ AGPL |

จุดที่ปัก v7.0 ไว้ — เปลี่ยนเมื่อไหร่ ใบอนุญาตของทั้งโปรเจกต์เปลี่ยนทันที

| ไฟล์ | บรรทัด |
|---|---|
| `Colab-Training/YoloV5-Training.ipynb` | `git clone -b v7.0 --depth 1 https://github.com/ultralytics/yolov5` |
| `Colab-Training/YoloV5-Training.ipynb` | `wget .../releases/download/v7.0/yolov5s.pt` (ดึง weights จาก release v7.0 โดยตรง) |
| `Integration-Camera-Local/Pylon+Yolov5/yolo_lib.py` | `torch.hub.load("ultralytics/yolov5:v7.0", ...)` |

> **หมายเหตุ** — โค้ดของ YOLOv5 **ไม่ได้ถูก commit ไว้ใน repo นี้** แต่ถูกดาวน์โหลดตอนรันครั้งแรก
> การแจกจ่าย repo นี้จึงไม่ใช่การ redistribute ซอร์ส GPL ของผู้อื่น แต่ผลรวมตอนใช้งานยังเป็น combined work

---

<a id="pylon-gpl"></a>

## 2 · GPL-3.0 กับ Basler pylon — ประเด็นที่ต้องรู้ก่อนแจกจ่าย

`pypylon` เป็น **BSD-3-Clause** แต่ทำงานไม่ได้ถ้าไม่มี **Basler pylon Camera Software Suite**
ซึ่งเป็นซอฟต์แวร์ **กรรมสิทธิ์ของ Basler** (PyPI จึงจัด wheel ของ pypylon เป็น `Other/Proprietary License`
เพราะ wheel มี binary ของ pylon runtime รวมมาด้วย)

ผลที่ตามมาสองข้อ

1. **ห้าม redistribute ตัว pylon runtime** ต่อ — ผู้ใช้ปลายทางต้องดาวน์โหลดและยอมรับ EULA ของ Basler เอง
   โปรเจกต์นี้จึงไม่ได้แนบไฟล์ติดตั้งของ Basler มาให้ มีแต่คำสั่งติดตั้ง `pypylon` ผ่าน pip
2. การนำโค้ด GPL-3.0 ไปเชื่อมกับไลบรารีกรรมสิทธิ์แล้ว **แจกจ่ายเป็นชุดเดียวกัน** เป็นประเด็นที่ GPL เข้มงวด
   ถ้าคุณจะแจกจ่ายผลิตภัณฑ์ที่รวมทั้งสองอย่าง ควรปรึกษาฝ่ายกฎหมาย

<details>
<summary>ถ้าต้องการอนุญาตให้ปลายทางเชื่อมกับ pylon ได้ — ข้อความ §7 ที่เอาไปใช้ได้ (<b>ยังไม่ได้ประกาศใช้ในโปรเจกต์นี้</b>)</summary>

GPL-3.0 มาตรา 7 เปิดให้เจ้าของลิขสิทธิ์เพิ่ม "additional permission" ได้
ถ้าต้องการปลดล็อกข้อ 2 ข้างบนให้คนที่นำไปพัฒนาต่อ ให้วางข้อความนี้ไว้ใน `LICENSE` **ต่อท้าย** ตัวบท GPL
หรือใส่เป็น header ในไฟล์ที่คุณเป็นเจ้าของลิขสิทธิ์

```text
Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with the Basler pylon Camera Software Suite (or a modified version of that
library), containing parts covered by the terms of the Basler End User License
Agreement, the licensors of this Program grant you additional permission to
convey the resulting work.
```

⚠ ข้อความนี้ครอบคลุมได้เฉพาะส่วนที่ **คุณเป็นเจ้าของลิขสิทธิ์เอง** เท่านั้น
ไม่สามารถให้สิทธิ์แทนผู้ถือลิขสิทธิ์ของ YOLOv5 ได้ — ตัดสินใจร่วมกับฝ่ายกฎหมายก่อนใช้

</details>

---

## 3 · ส่วนประกอบหลัก

| ส่วนประกอบ | ใบอนุญาต | ใช้ตรงไหน |
|---|---|---|
| [YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0) — Ultralytics | **GPL-3.0** | สถาปัตยกรรมโมเดล · `train.py` · `export.py` · โหลดผ่าน `torch.hub` |
| `best.pt` (weights ในโปรเจกต์นี้) | **GPL-3.0** (ตามต้นทาง) | `Integration-Camera-Local/Pylon+Yolov5/best.pt` |
| Pretrained `yolov5s.pt` / `yolov5m.pt` / `yolov5n.pt` (release v7.0) | **GPL-3.0** | จุดตั้งต้นการเทรนในโน้ตบุ๊ก |
| [pypylon](https://github.com/basler/pypylon) — Basler | BSD-3-Clause | `yolo_lib.open_camera()` · `candy_lib` · `stepN_camera.py` |
| **Basler pylon Camera Software Suite** | **กรรมสิทธิ์ — Basler EULA** | runtime ที่ pypylon เรียกใช้ · **ไม่ได้แจกจ่ายมากับ repo นี้** |
| [PyTorch](https://github.com/pytorch/pytorch) (`torch`) | BSD-3-Clause (ไฟล์ LICENSE รวมผู้ถือลิขสิทธิ์หลายราย)<br/>PyPI ระบุเป็น `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | inference engine |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | dependency ของ YOLOv5 |
| [OpenCV](https://github.com/opencv/opencv) | Apache-2.0 | ประมวลผลภาพทั้งหมด |
| [opencv-python](https://github.com/opencv/opencv-python) (ตัว wheel) | MIT | แพ็กเกจที่ pip ติดตั้ง |
| [NumPy](https://github.com/numpy/numpy) | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | ทั่วทั้งโปรเจกต์ |

## 4 · Dependency ของ YOLOv5 v7.0 (ติดตั้งผ่าน `requirements-yolo.txt`)

| แพ็กเกจ | ใบอนุญาต |
|---|---|
| Pillow | MIT-CMU (HPND) |
| matplotlib | Matplotlib License (BSD-compatible, PSF-based) |
| pandas · scipy · seaborn · psutil · GitPython · IPython | BSD-3-Clause |
| PyYAML · thop | MIT |
| requests | Apache-2.0 |
| tqdm | `MPL-2.0 AND MIT` |

<sub>ข้อมูลใบอนุญาตดึงจาก metadata บน PyPI และไฟล์ LICENSE ของแต่ละ repo เมื่อ 29 ส.ค. 2026
เวอร์ชันที่ติดตั้งจริงอาจต่างออกไป — ตรวจซ้ำด้วย `pip-licenses` ก่อนแจกจ่ายผลิตภัณฑ์</sub>

---

## 5 · ข้อมูลและสื่อ

| รายการ | ใบอนุญาต / เงื่อนไข |
|---|---|
| Dataset [`typecandy`](https://universe.roboflow.com/kmitl-kmitl/typecandy/dataset/1) (Roboflow Universe, workspace `kmitl-kmitl`) | **CC BY 4.0** — ต้องระบุที่มาเมื่อนำไปใช้หรือดัดแปลง |
| `Dataset/Roboflow/Candy-factory.v1i.yolov5pytorch.zip` | CC BY 4.0 (export ของ dataset ข้างต้น) |
| `Dataset/Image_Origin/` · `Dataset/Video_Origin/` · `Integration-Camera-Local/Candy.png` | ถ่ายเองในโปรเจกต์นี้ — GPL-3.0 ตามโปรเจกต์ |
| ภาพใน `docs/` | ถ่าย/สร้างเองในโปรเจกต์นี้ — GPL-3.0 ตามโปรเจกต์ |

**ข้อความ attribution ที่ใช้ได้ตามเงื่อนไข CC BY 4.0**

```text
"typecandy" by kmitl-kmitl, via Roboflow Universe
https://universe.roboflow.com/kmitl-kmitl/typecandy/dataset/1
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
```

<sub>ยี่ห้อลูกอมที่ปรากฏในภาพ (Clorets, Halls, Skittles) เป็นเครื่องหมายการค้าของเจ้าของแต่ละราย
ใช้ในโปรเจกต์นี้เพื่อการอ้างอิงเชิงเทคนิคเท่านั้น ไม่ได้แสดงถึงความเกี่ยวข้องหรือการรับรองใด ๆ</sub>

---

## 6 · เช็คลิสต์สำหรับคนที่นำไปพัฒนาต่อ

- [ ] เก็บไฟล์ [`LICENSE`](LICENSE) และไฟล์นี้ไว้ในงานที่แจกจ่าย
- [ ] ระบุว่าดัดแปลงจากโปรเจกต์นี้ พร้อมวันที่ที่แก้ไข (GPL-3.0 §5a)
- [ ] เผยแพร่ซอร์สของส่วนที่ดัดแปลงภายใต้ GPL-3.0 เมื่อ **แจกจ่าย** (ใช้ภายในองค์กรอย่างเดียวไม่ต้อง)
- [ ] คงคำระบุที่มาของ dataset CC BY 4.0 ไว้ ถ้ายังใช้ข้อมูลชุดเดิม
- [ ] ตรวจว่ายังปักที่ YOLOv5 `v7.0` — ถ้าอัปเป็น branch ปัจจุบัน โปรเจกต์จะกลายเป็น **AGPL-3.0**
- [ ] เทรนโมเดลใหม่แล้ว ให้ปรับ [`README.md`](README.md#modelcard) และระบุที่มาของ dataset ชุดใหม่
- [ ] ห้ามแนบ installer ของ Basler pylon ไปกับงานที่แจกจ่าย

---

<sub>เอกสารนี้เป็นการสรุปเพื่อความเข้าใจทางวิศวกรรม **ไม่ใช่คำแนะนำทางกฎหมาย**
การนำไปใช้ในผลิตภัณฑ์เชิงพาณิชย์ควรผ่านการตรวจสอบจากฝ่ายกฎหมายขององค์กร</sub>
