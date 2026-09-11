# walking-fly

초파리 전신 물리 모델([flybody](https://github.com/TuragaLab/flybody), Google DeepMind × HHMI Janelia)이
평소 6족 자세에서 **스스로 일어나 뒷다리 두 개로 목표 지점까지 걸어가도록** 강화학습(PPO)으로 훈련합니다.
모든 관절은 자유롭고, "2족 보행"은 보상으로만 유도합니다. 넘어지면 스스로 일어나야 합니다.

- 물리: MuJoCo 3.13 + MJX(MuJoCo Warp, GPU 병렬 4096 환경)
- 학습: Brax PPO, 자동 커리큘럼(걷기 → 일어서기 → 보조력 제거)
- 관측: 실제 로봇이 가질 수 있는 센서만(관절 엔코더, IMU, 발끝 힘·촉각, 눈에 보이는 목표 방향)
- 뷰어: 브라우저 안에서 MuJoCo WASM 물리 + 정책 추론이 실시간으로 돕니다

## 빠른 시작

```bash
git clone https://github.com/rocknroll17/walking-fly.git && cd walking-fly
bash scripts/setup.sh          # 환경 점검, Python 3.12 venv(uv), flybody 모델, 웹 자산 빌드 (~10분)
bash scripts/run.sh start      # 학습(GPU) + 체크포인트 감시(CPU) + 웹 서버 시작
```

- 진행 페이지: `http://<서버IP>:8765/` (학습 곡선, 체크포인트별 클립, 설계 근거)
- 실시간 뷰어: `http://<서버IP>:8765/viewer.html` (2분마다 최신 정책 자동 반영)
- 상태/중지/로그: `bash scripts/run.sh status|stop|logs`
- GPU 없이 뷰어만: `bash scripts/setup.sh --viewer` 후 `cd web && python3 -m http.server 8765`

요구 사항: Linux, NVIDIA GPU(Volta 이상, 드라이버 ≥ 525, 8 GB VRAM 이상), Node.js ≥ 18(뷰어), 인터넷(최초 설치).
EGL 헤드리스 렌더링이 없는 머신에서는 `MUJOCO_GL=osmesa bash scripts/run.sh start` 로 클립 렌더를 대체할 수 있습니다.

## 다른 서버로 옮겨서 이어 학습하기

```bash
# 원래 서버에서: 최신 체크포인트 + 커리큘럼 단계 + 누적 스텝을 한 파일로
bash scripts/migrate.sh pack            # -> runs/bundle.tar.gz (수 MB)
# 새 서버에서:
git clone https://github.com/rocknroll17/walking-fly.git && cd walking-fly && bash scripts/setup.sh
bash scripts/run.sh start --resume bundle.tar.gz
```

## 구조

```
flybiped/
  model.py        flybody 자산에서 2족 학습용 MJCF 생성 (입·더듬이·평형곤 고정, 날개 서보화, 목표 마커)
  pose.py         6족 자세와 뒷다리 서기 자세 탐색 (build/stand_pose.json)
  env.py          MJX 환경: 센서 전용 관측, 보행 판정, 보상, 시작 자세(6족/서기/낙하/뒤집힘), 밀치기, 보조력
  train.py        Brax PPO 학습 (GPU), 체크포인트 저장
  autopilot.py    자동 커리큘럼: 걷기 → 일어서기 집중 훈련 → 혼합 + 보조력 감소 → 완료 판정
  watch_export.py 체크포인트 → 뷰어용 policy.json, CPU 평가(시작 자세별·보행 형태), 3초 클립
  evaluate.py     numpy 재구현 환경 (CPU 평가/영상용)
  export.py       정책 + 환경 상수 내보내기
  web_assets.py   뷰어용 경량 메시 생성
web/
  viewer.html, src/viewer.js   MuJoCo WASM + three.js 실시간 뷰어 (관측·행동 코드는 env.py와 동일)
  index.html                   진행 상황·설계 근거 페이지
scripts/setup.sh, scripts/run.sh
```

## 설계 근거 (요약)

- 2족 강제: 앞다리 접촉 벌점 + 전진 보상은 "뒷다리로 서서 한 발 이상 접지"일 때만 (TumblerNet npj Robotics 2025, CyberDog2 2023)
- 일어서기: 넘어져도 종료하지 않고 벌점만, 낙하·뒤집힘 시작 섞기, 위로 당기는 보조력을 단계적으로 0으로 (MuJoCo Playground Go1 Getup, HoST 2025)
- 보행 형태: 두 발 동시 체공 벌점, 교대 딛기 보상 (깡충 걸음 방지)
- 뒤집힘 복구: 넘어진 동안 날개 자유(곤충은 날개로 바닥을 밀어 일어남, JHU Terradynamics 2019), 전 구간 기울기가 있는 자세 보상
- 제어 4 ms / 물리 0.4 ms, 할인율 0.996 (γ = 1 − Δt/T), 관절 속도 제한 60 rad/s (실제 초파리 걸음 주기 110 ms)

자세한 근거와 수치는 진행 페이지의 "설계 근거" 섹션에 있습니다.

## 라이선스

MIT. 초파리 모델과 메시는 설치 시 내려받는 flybody(Apache-2.0)의 것입니다.
