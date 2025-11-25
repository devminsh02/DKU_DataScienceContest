2025년에 이루어진 단국대학교 데이터 사이언스 경진대회 관련 코드입니다.

해당 코드로 우수상을 수상하였습니다. (Linked IN에 자세한 내용이 있습니다)

대회의 목표는 24년도 12월부터 25년4월까지의 전기 사용량 관련 데이터를 주고, 25년 5월의 전기 사용량을 예측하는 대회입니다.

데이터셋은 공유금지 규약이 있어, 공개하지 않았으며 코드 입출력 path도 실제 경로가 아닌 path로 작성되어 있습니다.

데이터는 
Module, timestamp, local, operation, volageR, voltageS, voltageT, voltageRS, voltageST, voltageTR, current, currentS, currentT, ActivePower, PowerFactorR, PowerFactorS, PowerFactorT, reactivePowerLagging, accumActiveEnergy
로 총 12개로 구성되어 있습니다.
또한 module은 총 18개로 각각 위의 사용량을 전부 가지고 있습니다. 

코드의 전체 흐름은 다음과 같습니다.
1)seperate.py로 각 module별로 분해
2)노이즈 제거
2.1)이론적 전력값과 잔차를 비교하여 threshold값 설정 - 식은 삼상 교류 전력 공식에 근거함
2.2)XGBoost.py 파일로 이상치를 탐지할경우 이상치를 XGBoost모델을 통해 보정함
3)학습진행
3.1)PatchTST.py로 5월달 전력 예측 진행
4)후처리
4.1)merged_activePower.py 파일로 1분당 activepower를 1시간 단위로 묶고 같은 시간의 모든 module의 activepower를 합산
4.2)merged_activePower_1hour.py 파일로 대회의 출제 양식에 맞게 변형함(1시간 단위 병합)

