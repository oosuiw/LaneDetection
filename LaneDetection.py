import cv2
import numpy as np
import time
import os

# 전역 변수로 파라미터 설정
class Params:
    # ROI 파라미터
    roi_top_width = 45  # ROI 상단 너비 (%)
    roi_bottom_width = 0  # ROI 하단 너비 (%)
    roi_height = 55  # ROI 높이 (%)
    
    # HSV 임계값
    white_value = 150  # 흰색 차선 Value
    white_saturation = 40  # 흰색 차선 Saturation
    yellow_value = 100  # 노란색 차선 Value
    yellow_saturation = 40  # 노란색 차선 Saturation
    
    # 차선 검출 파라미터
    min_slope = 0.1  # 최소 기울기
    max_slope = 1.2  # 최대 기울기

def create_control_window():
    cv2.namedWindow('Controls')
    
    # ROI 컨트롤
    cv2.createTrackbar('ROI Top Width %', 'Controls', Params.roi_top_width, 100, 
                      lambda x: setattr(Params, 'roi_top_width', x))
    cv2.createTrackbar('ROI Bottom Width %', 'Controls', Params.roi_bottom_width, 100,
                      lambda x: setattr(Params, 'roi_bottom_width', x))
    cv2.createTrackbar('ROI Height %', 'Controls', Params.roi_height, 100,
                      lambda x: setattr(Params, 'roi_height', x))
    
    # HSV 임계값 컨트롤
    cv2.createTrackbar('White Value', 'Controls', Params.white_value, 255,
                      lambda x: setattr(Params, 'white_value', x))
    cv2.createTrackbar('White Saturation', 'Controls', Params.white_saturation, 255,
                      lambda x: setattr(Params, 'white_saturation', x))
    cv2.createTrackbar('Yellow Value', 'Controls', Params.yellow_value, 255,
                      lambda x: setattr(Params, 'yellow_value', x))
    cv2.createTrackbar('Yellow Saturation', 'Controls', Params.yellow_saturation, 255,
                      lambda x: setattr(Params, 'yellow_saturation', x))
    
    # 차선 검출 파라미터 컨트롤 (기울기 * 100)
    cv2.createTrackbar('Min Slope x100', 'Controls', int(Params.min_slope * 100), 200,
                      lambda x: setattr(Params, 'min_slope', x / 100))
    cv2.createTrackbar('Max Slope x100', 'Controls', int(Params.max_slope * 100), 200,
                      lambda x: setattr(Params, 'max_slope', x / 100))

def process_frame(frame):
    # 이미지 전처리
    height, width = frame.shape[:2]
    
    # 이미지 밝기 개선
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    
    # HSV 색공간으로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 흰색 차선 검출을 위한 임계값 설정
    white_lower = np.array([0, 0, Params.white_value])
    white_upper = np.array([180, Params.white_saturation, 255])
    
    # 노란색 차선 검출을 위한 임계값 설정
    yellow_lower = np.array([15, Params.yellow_saturation, Params.yellow_value])
    yellow_upper = np.array([35, 255, 255])
    
    # 흰색과 노란색 차선 마스크 생성
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # 마스크 합치기
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 동적 ROI 생성
    roi_points = np.array([
        [int(width * Params.roi_bottom_width/100), height],             # 왼쪽 하단
        [int(width * (50 - Params.roi_top_width/2)/100), int(height * (100-Params.roi_height)/100)],  # 왼쪽 상단
        [int(width * (50 + Params.roi_top_width/2)/100), int(height * (100-Params.roi_height)/100)],  # 오른쪽 상단
        [int(width * (100-Params.roi_bottom_width)/100), height]              # 오른쪽 하단
    ], dtype=np.int32)
    
    # ROI 마스크 생성
    roi_mask = np.zeros_like(combined_mask)
    cv2.fillPoly(roi_mask, [roi_points], 255)
    combined_mask = cv2.bitwise_and(combined_mask, roi_mask)
    
    # 엣지 검출
    edges = cv2.Canny(combined_mask, 30, 100)
    
    # 허프 변환으로 직선 검출
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,       
        minLineLength=20,
        maxLineGap=100
    )
    
    # 차로 영역 시각화를 위한 마스크
    lane_mask = np.zeros_like(frame)
    debug_image = frame.copy()
    
    if lines is not None:
        # 왼쪽/오른쪽 차선 구분을 위한 리스트
        left_points = []
        right_points = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 수직에 가까운 선은 제외
            if abs(x2 - x1) < 1:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # 동적 기울기 범위 적용
            if -Params.max_slope < slope < -Params.min_slope and x1 < width * 0.6:  # 왼쪽 차선
                left_points.append([x1, y1])
                left_points.append([x2, y2])
            elif Params.min_slope < slope < Params.max_slope and x1 > width * 0.4:  # 오른쪽 차선
                right_points.append([x1, y1])
                right_points.append([x2, y2])
            
            # 검출된 모든 선 시각화
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # 차선 피팅 및 시각화
        if len(left_points) > 0 or len(right_points) > 0:
            try:
                # y 좌표 범위 생성
                y_points = np.linspace(height * (100-Params.roi_height)/100, height, num=30, dtype=np.int32)
                left_x = None
                right_x = None
                
                # 왼쪽 차선 곡선 피팅
                if len(left_points) >= 6:
                    left_points = np.array(left_points)
                    left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
                    left_x = np.polyval(left_fit, y_points)
                    
                    if len(right_points) < 6:
                        right_x = left_x + width * 0.45
                
                # 오른쪽 차선 곡선 피팅
                if len(right_points) >= 6:
                    right_points = np.array(right_points)
                    right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
                    right_x = np.polyval(right_fit, y_points)
                    
                    if len(left_points) < 6:
                        left_x = right_x - width * 0.45

                # 차로 영역 포인트 생성
                if left_x is not None and right_x is not None:
                    lane_points = []
                    for i in range(len(y_points)):
                        lane_points.append([int(left_x[i]), int(y_points[i])])
                    for i in range(len(y_points)-1, -1, -1):
                        lane_points.append([int(right_x[i]), int(y_points[i])])
                    lane_points = np.array(lane_points, dtype=np.int32)

                    # 차로 영역 채우기
                    cv2.fillPoly(lane_mask, [lane_points], (255, 144, 30))

                    # 중앙선 그리기
                    for i in range(len(y_points)-1):
                        center_x1 = int((left_x[i] + right_x[i]) / 2)
                        center_x2 = int((left_x[i+1] + right_x[i+1]) / 2)
                        center_y1 = int(y_points[i])
                        center_y2 = int(y_points[i+1])
                        cv2.line(lane_mask, (center_x1, center_y1),
                                (center_x2, center_y2), (0, 255, 0), 3)

            except np.linalg.LinAlgError:
                pass
    
    # ROI 영역 시각화
    cv2.polylines(debug_image, [roi_points], True, (0, 0, 255), 2)
    
    # 디버깅을 위한 이미지 표시
    debug_view = np.zeros((height, width * 2, 3), dtype=np.uint8)
    debug_view[:, :width] = debug_image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    debug_view[:, width:] = edges_colored
    
    # 원본 이미지와 차로 마스크 합성
    result = cv2.addWeighted(frame, 1, lane_mask, 0.3, 0)
    
    return result, debug_view

def find_camera():
    # /dev/video* 장치 확인
    video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
    if not video_devices:
        print("카메라 장치가 감지되지 않습니다.")
        return -1
    
    print("감지된 카메라 장치:")
    for device in video_devices:
        print(f"/dev/{device}")
    
    # 각 장치 시도
    for device in video_devices:
        device_path = f"/dev/{device}"
        print(f"\n{device_path} 시도 중...")
        try:
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            
            if not cap.isOpened():
                print(f"{device_path} 열 수 없습니다.")
                continue
                
            print(f"{device_path} 열기 성공")
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("카메라 설정 시도 중...")
            
            for i in range(10):
                print(f"프레임 읽기 시도 {i+1}/10...")
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"{device_path} 연결 성공!")
                    print(f"프레임 크기: {frame.shape}")
                    cap.release()
                    return device_path
                time.sleep(0.2)
            
            print(f"{device_path} 프레임 읽기 실패")
            cap.release()
            
        except Exception as e:
            print(f"카메라 {device_path} 연결 중 오류 발생: {str(e)}")
            continue
    
    return -1

def main():
    print("웹캠을 초기화하는 중...")
    
    # 컨트롤 윈도우 생성
    create_control_window()
    
    # 사용 가능한 카메라 찾기
    camera_path = find_camera()
    if camera_path == -1:
        print("사용 가능한 카메라를 찾을 수 없습니다.")
        return
    
    cap = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 다른 카메라를 사용해보세요.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n웹캠이 성공적으로 연결되었습니다.")
    print(f"카메라 경로: {camera_path}")
    print(f"해상도: {width}x{height}")
    print(f"FPS: {fps}")
    print("'q' 키를 누르면 프로그램이 종료됩니다.")
    
    time.sleep(1)
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("프레임을 읽을 수 없습니다. 카메라를 확인해주세요.")
                break
                
            processed_frame, debug_image = process_frame(frame)
            
            cv2.imshow('Lane Detection', processed_frame)
            cv2.imshow('Debug View', debug_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {str(e)}")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()
