# 🚗 Lane Detection with OpenCV

![Lane Detection Demo]
*자율주행 차선 검출 시스템의 강력한 데모*

---

## 📖 프로젝트 개요

**Lane Detection with OpenCV**는 실시간 차선 검출을 위한 Python 기반 프로젝트입니다. OpenCV를 활용해 웹캠 입력을 처리하며, HSV 색상 공간과 허프 변환을 통해 흰색과 노란색 차선을 정확히 검출합니다. 동적 ROI 설정, 직관적인 트랙바 인터페이스, 그리고 실시간 디버깅 뷰를 제공하여 자율주행 연구 및 개발에 최적화된 도구입니다.

이 프로젝트는 ROS2 Humble 및 Autoware Universe 환경에서 테스트되었으며, Ubuntu 22.04 LTS에서 안정적으로 동작합니다. 초보자부터 전문가까지 쉽게 커스터마이징할 수 있는 유연한 구조를 자랑합니다.

---

## ✨ 주요 기능

- **실시간 차선 검출**: 흰색과 노란색 차선을 HSV 색상 공간을 통해 정확히 분리
- **동적 ROI 설정**: 트랙바로 ROI의 상단/하단 너비와 높이를 실시간 조정 가능
- **직관적인 디버깅 뷰**: 원본 이미지와 Canny 엣지 결과를 나란히 표시
- **유연한 파라미터 조정**: 차선 기울기, HSV 임계값 등을 트랙바로 즉시 튜닝
- **카메라 자동 감지**: `/dev/video*` 장치를 자동으로 탐지해 연결
- **차로 영역 시각화**: 검출된 차선을 기반으로 주행 가능 영역을 색상으로 표시

---

## 🛠 설치 방법

### 1. 사전 요구사항
- **운영체제**: Ubuntu 22.04 LTS
- **Python**: 3.8 이상
- **라이브러리**:
  ```bash
  pip install opencv-python numpy
  ```
- **카메라**: USB 웹캠 또는 시스템에 연결된 카메라 장치
- **선택사항**: ROS2 Humble 및 Autoware Universe (v1.0) 환경 (ROS2 Wrapping 및 autoware.universe(release/v1.0)를 활용할 예정 | 추가로 원활한 실행을 위해 C++ 포팅 또한 예정)

### 2. 레포지토리 클론
```bash
git clone https://github.com/yourusername/lane-detection-opencv.git
cd lane-detection-opencv
```

### 3. 실행
```bash
python3 lane_detection.py
```

> **참고**: 실행 전 웹캠이 `/dev/video*`로 인식되는지 확인하세요. 프로그램은 자동으로 사용 가능한 카메라를 탐지합니다.

---

## 🎮 사용 방법

1. 프로그램 실행 후 **Controls** 창에서 트랙바를 사용해 파라미터를 조정:
   - **ROI Top Width/Bottom Width/Height**: 차선 검출 영역 설정
   - **White/Yellow Value/Saturation**: 차선 색상 임계값 튜닝
   - **Min/Max Slope**: 차선 기울기 필터링
2. **Lane Detection** 창에서 실시간 차선 검출 결과 확인
3. **Debug View** 창에서 원본 이미지와 엣지 검출 결과 비교
4. 종료하려면 `q` 키를 누르세요.

---

## 📸 실행 화면

| **Lane Detection 결과** | **Debug View** |
|-------------------------|----------------|
| ![Result](https://via.placeholder.com/400x300.png?text=Lane+Detection+Result) | ![Debug](https://via.placeholder.com/400x300.png?text=Debug+View) |

---

## 🧠 코드 구조

```plaintext
lane_detection.py
├── Params: 파라미터 클래스 (ROI, HSV, 기울기 설정)
├── create_control_window(): 트랙바 인터페이스 생성
├── process_frame(): 프레임 처리 및 차선 검출 핵심 로직
├── find_camera(): 카메라 장치 자동 탐지
├── main(): 프로그램 실행 흐름 관리
```

### 주요 알고리즘
1. **HSV 색상 필터링**: 흰색/노란색 차선 분리
2. **동적 ROI 적용**: 사다리꼴 ROI로 검출 영역 제한
3. **Canny 엣지 + 허프 변환**: 직선 검출
4. **2차 곡선 피팅**: 안정적인 차선 추적
5. **차로 시각화**: 주행 가능 영역 및 중앙선 표시

---

## 🔧 커스터마이징 팁

- **ROI 최적화**: `Params.roi_*` 값을 조정해 카메라 시야에 맞는 영역 설정
- **색상 튜닝**: 조명 조건에 따라 `white_value`, `yellow_saturation` 등 수정
- **성능 향상**: `HoughLinesP` 파라미터(`threshold`, `minLineLength`) 조정
- **ROS2 통합**: Python 노드로 변환해 Autoware Universe와 연동 가능

---

## ⚠️ 알려진 이슈

- **카메라 호환성**: 일부 웹캠은 V4L2 드라이버를 지원하지 않을 수 있음
- **조명 민감도**: 강한 역광에서는 HSV 임계값 재조정 필요
- **성능**: 고해상도 영상에서 프레임 드롭 발생 가능 (640x480 권장)

---

## 🌟 기여 방법

이 프로젝트는 오픈소스로 유지됩니다! 다음과 같은 방법으로 기여할 수 있습니다:
1. **버그 리포트**: 이슈 템플릿을 사용해 문제 제보
2. **기능 제안**: 새로운 아이디어 공유
3. **풀 리퀘스트**: 코드 개선 및 최적화 제안

```bash
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```

---

## 🙌 감사의 말

- **OpenCV 커뮤니티**: 강력한 컴퓨터 비전 라이브러리 제공
- **ROS2 & Autoware 팀**: 자율주행 개발 환경 지원
- **당신**: 이 프로젝트를 확인해 준 모든 분들!

---

*더 멋진 차선 검출을 위해, 지금 바로 시작하세요!* 🚀