# 복합적인 교통상황에서의 ACC(Adaptive Cruise Control) 구현

## 프로젝트 목표
![image](https://github.com/SeungJiRyu/Embedded-ACC-Project/assets/108774002/f9b6e128-4cea-4c59-b6e8-bcbc8cc6fc3e)


1. 기존의 task를 기반으로 교차로에서 신호등 구간과 버스 정류장, 도로를 가로지르는 무단횡단 상황을 추가한 복합적인 교통상황에서도 정해진 task를 수행하도록 하는 것이 첫번째 목표입니다.
- 신호등 : 빨간불(정지), 초록불(정상주행)
- 버스정류장 : 사람이 있을 때(정차), 사람이 없는 경우(정차하지 않고 주행) 
- 무단횡단 보행자 인식 시 급정거 혹은 회피주행

2. ACC 기능을 구현하여 도로 주행 중 선행차량이 존재하는 경우에는 ACC 기능이 자동으로 켜져 적정 거리를 유지하면서 선행 차량을 추종하고, 선행 차량이 감지되지 않을 시에는 주변환경에 맞게 기존에 설정한 task를 수행하도록 하는 것이 두번째 목표입니다.
  
![image](https://github.com/SeungJiRyu/Embedded-ACC-Project/assets/108774002/8bac0e18-5689-48f8-ab34-e82ed422939c)


