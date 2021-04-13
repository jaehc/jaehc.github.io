---
title: "[아티클 리뷰] Active Learning Tutorial"
excerpt: "Active Learning에 관한 좋은 글을 찾아서 주요 포인트를 요약했습니다."

categories:
    - 기타
tags:
    - active learning
    - semi-supervised learning
---

<br/>

이번 글에서는 Active Learning에 대한 좋은 글 하나를 소개하려 합니다.  

본격적으로 들어가기에 앞서서 현업에서 ML을 적용할 때 발생하는 고충에 대한 얘기로 시작하려 합니다. (물론 제 주관적인 견해이기 때문에 일반화되지 않습니다.)

현업에서 ML을 적용하려면 학습 데이터와 관련하여 문제가 되는 상황은 다음 중 하나가 되는 것 같아요. 

1. ML학습에 필요한 데이터가 없다.
2. 데이터는 있는데 레이블이 달려 있지 않다. 
3. 데이터도 있고 레이블이 있는데, 레이블에 노이즈가 있다. 즉, 레이블이 부정확하다. 

1번은 갈 길이 좀 먼데, 문제 정의부터 시작해서 데이터 수집을 시작해야 하겠네요. 2번이나 3번 상황도 많을 텐데요, 특히나 2번의 경우가 많습니다. 데이터가 굉장히 많긴 한데, 레이블이 없어서 ML 학습을 하지 못하는 경우입니다. 

풀고자 하는 문제가 개와 고양이를 구분하거나 새소리, 고양이 소리를 구분하는 일반적인 도메인에 있고 예산이 넉넉하다면, 클라우드 소싱으로도 레이블링 작업을 할 수도 있겠죠. 오픈된 데이터셋을 잘 활용할 수도 있겠고, 혹은 직접..

한편으로 일부 사례들은 위와 같이 좋은 상황도 아닙니다. 레이블을 달려면 전문지식을 갖춘 전문가의 도움이 필요한 분야가 있습니다. 예를 들어, 환자의 MRI 결과를 보고 암이냐 아니냐 판정하는 경우엔 일반인은 레이블을 달 수 없고, 의료 지식이 갖춘 전문가의 판단이 필수적입니다. 

아시다시피 이러한 전문가들은 몸값이 매우 비싸고, 아주 바쁘죠. 그 때문에 레이블링 비용이 기하급수적으로 증가합니다. 레이블링 비용 문제를 차치하더라도 이러한 적절한 전문가들을 확보하기도 상당히 어렵습니다.

비단 특수한 전문적인 분야가 아니더라도, 다소 일반적인 분야에서도 도메인 전문가가 바쁘다든지, 비협조적이라든지, 어떠한 이유에서든 다량의 레이블된 데이터를 확보하기 어려운 경우가 많습니다. 

최근 들어 IT분야 이외에 여러 산업에 AI와 ML 기술이 적용되기 시작하면서 이러한 레이블링 문제가 부각되었는데, 이 문제를 해결할 수 있는 대안으로 꼽히는 것 중 하나가 Active Learning입니다. (Active Learning 자체는 오래 연구된 분야더군요.) Active Learning은 레이블링을 할 때 ML의 도움을 받아서 좀 더 효율적으로 레이블링을 하는 것을 추구합니다. 일종의 labeling efficiency죠. 주어진 레이블링 예산에서 최대한 효율적으로 레이블을 해서 성능을 높이는 것이 목표입니다. 

관련해서 논문도 많이 나오고 아티클도 많지만, 그런 것들은 천천히 리뷰해보고, 우선은 그중에 좀 짧고 내용이 알찬 글을 발견해서 주요 포인트를 요약해 볼까 합니다. 


# Active Learning, part1: the Theory

먼저 본 글의 출처를 밝힙니다. 

[Active Learning, part1: the Theory](https://blog.scaleway.com/active-learning-some-datapoints-are-more-equal-than-others/)

[Active Learning, part2: the Practice](https://blog.scaleway.com/active-learning-some-datapoints-are-more-equal-than-others/)

## Active learning: when and why do we need it?
* Active learning은 두 가지 문제에 대한 해결법임, 레이블 데이터의 양적인 문제와 질적인 문제.
* 먼저, 레이블 데이터의 양적인 문제는 레이블된 데이터를 확보하기 어렵다는 것인데, semi-supervised learning으로 이를 완화할 수 있음.
  * semi-supervised learning은 unlabeled data를 이용하여 supervised learning의 성능을 올리는 것임.
  * 따라서 전체 데이터에서 어떤 샘플에 레이블이 있는지가 성능을 결정함. 좀 더 representative하거나 informative하면 더 좋다고 함.
  * 이러한 샘플을 가려내기 위해서 ML model 자체를 이용하고자 하는 것이 Active Learning. 
* 그다음으로, 레이블 데이터의 질적인 문제는 트레이닝 데이터의 품질 이슈임.
  * 레이블에 노이즈가 일부 끼어 있거나, 데이터에 이상치가 존재할 경우에는 데이터가 많다고 성능이 올라가는 것이 아님.
  * semi-supervised setting에서만 국한된 것이 아니고, supervised learning에서 일반적으로 발생하는 문제임.
  * Active learning은 이러한 unhelpful data의 중요도를 낮춰서 모델의 성능을 올릴 수 있음. 

## Active learning: what is it?
* Human-in-the-Loop (HITL) ML의 하위 분야로 ML 학습 시에 사람의 역할이 강조됨. 
* ML 학습 과정은 다음 과정을 통해서 일어남.  
  1. oracle: 모델 학습시에 ground truth 레이블을 공급하는 역할인데, 주로 특정 분야의 전문가를 말함.
  2. 모델은 레이블이 있는 일부 데이터에 대해서 학습, 혹은 이미 보유하거나 알려진 데이터를 이용해서 학습함. 이때 모델의 성능은 초기에 확보한 validation set을 통해서 평가할 수 있음.
  3. 학습을 완료한 모델이 레이블이 없는 샘플을 분류하거나 예측하고, 이 중에서 모델 예측의 불활실성이 높거나, 그 외 다른 평가척도를 기준으로 oracle에게 레이블링 요청을 함.
  4. 1-3번 과정을 반복적으로 수행하는데, 모델이 원하는 성능에 도달하거나, 레이블링에 할당된 예산을 모두 소진할 때까지 반복함. 
* <span style="color:red">(코멘트) Active Learning 방식을 적용할 때 사용하는 모델은 supervised ML이기만 하면 어떠한 모델이라도 상관없지만, 좀 더 정확하게는 score나 confidence, 확률 등의 수치형 결과를 반환하는 모델이어야 함.</span> 
* Active Learning이 아닌 방식은 이와 비교하기 위해서 Passive Learning이라고 부르는데, Active Learning은 학습 중인 ML로부터 받은 피드백을 기반으로 다음 학습 샘플을 선택하는 것에 반해서, Active Learning은 학습 샘플을 램덤으로 선택하거나, 혹은 모델 개발자로부터 주어짐.

## Active learning: how is it done?
* Active Learning을 통해서 레이블링이 필요한 샘플을 효과적으로 고를 수 있는데, 그러면 최초에 모델은 어떻게 학습시켜야 할까?
  1. Transfer learning: 풀고자 하는 문제와 데이터에 적용할 수 있는 Pre-train 모델이 있다면 이를 활용. 
  2. Random query: 램덤으로 일부 학습 샘플을 선택해서 레이블링함. 
  3. Random query를 통해서 일부 레이블 샘플을 확보하고, Pre-train 모델을 Fine-tunning함.
  4. Clustering한 후에 각 클러스터에서 샘플링함. 데이터에 몇 개의 클래스가 있는 지 모를 때 유용함. 
* <span style="color:red">(코멘트) 1번은 pre-train모델과 풀고자 하는 task가 일치 및 데이터 분포도 일치해야 가능함. 3번과 4번을 통해서 일부 샘플을 레이블링한 후에 1번을 이용하는 방식이 유용할 것임. 4번의 경우는 클러스터링을 위해서는 데이터 간의 거리 (metric)을 어떻게 계산할 것인지가 결정되어야 함. 만약 거리 함수를 근사치라도 알 수 있으면 풀고자 하는 문제는 굉장히 쉬워지나 그럴 일이 거의 없음. 물론 일부 케이스에서는 Euclidean distance와 같은 객관적인 metric을 사용할 수도 있음.</span>


## Streaming or Pooling?
* Streaming 방식은 샘플 하나씩 prediction한 후, 그 결과를 oracle에 보낼지 말지 결정하는 것인데, 예측한 class의 confidence가 특정 임계점보다 낮을 때 oracle로 보냄. 
* Pooling 방식은 모든 unlabeled data을 모두 prediction을 하고 난 뒤에, query strategy에 따라서 랭킹을 메긴 후 레이블링이 필요한 샘플을 선택함.
* <span style="color:red">(코멘트) 대부분 Pooling 방식으로 채택할 것인데, 유스케이스에 따라서 Streaming 방식도 유용할 것임.</span>

## Which query strategy?
* Uncertainty sampling이 가장 많이 사용됨. 간단하고 계산량도 적으며 성능도 괜찮음.
* Uncertainty sampling: 모델이 가장 불확실해하는 샘플을 선택하는 것인데, 불확실성을 어떻게 정의하느냐에 따라서 다음 세 가지로 나뉨.
  
  * Least confidence: 모델이 예측한 class의 confidence가 가장 낮은 샘플을 선택함. 이 방식에 문제점이 있는데 outlier나 데이터에 있는 noise 등을 선택할 가능성이 높음. (즉, Decision Boundary를 찾는 데는 도움이 안 됨) 
  * Margin sampling: Multiclass 분류에서 probability 기준으로 상위 1, 2등의 차이가 작은 것을 선택함. Binary class의 경우에는 Least confidence과 같은 결과일 것임 (h(x)=1/2 근처인 x)
  * Entropy sampling: Multiclass 분류에서 각 class에 소속할 확률 분포의 Entropy가 높은 것을 선택함.

* Query-by-committee: 여러 서로 다른 모델들을 학습 및 unlabled data에 예측을 수행하는데, 모델간의 예측 결과가 가장 많이 다른 샘플을 선택함. <span style="color:red">(코멘트) Ensemble 방식으로 별도의 모델들이 전체 샘플 공간의 복잡한 부분공간들을 효과적으로 탐색할 수 있어서 Active Learning의 아이디어와도 잘 어울림.</span>
* Expected model change: 어떤 샘플의 레이블을 알고 있다고 가정하고 학습할 때, loss 함수의 gradients의 기대값중 가장 큰 것을 고름. 즉, 샘플이 가질 수 있는 각 class에 대해 각각 loss 함수의 gradients를 구하고, 그 값들을 평균한 것임. <span style="color:red">(코멘트) Unlabeled 샘플의 수가 M이고, class의 수가 C면, M*C의 prediction을 하고 gradients를 구해야 함. 계산량이 너무 많기 때문에 Approximation을 하거나, 다른 앞선 기준을 적용하여 필터링해야 할 것임.</span>  
* Expected error reduction: 어떤 샘플을 레이블을 알고 있을 때, 샘플이 가질 수 있는 class 각각의 validation error를 구하고 평균함. 평균 validation error가 가장 큰 샘플을 선택. <span style="color:red">(코멘트) 데이터가 아주 작을 때 시도해 볼 만한데, 그러면 그냥 모두 레이블링하는 게 더 낫지 않을까?</span>

## Active learning: what could go wrong?
* Least confidence의 경우, 위에 언급했듯이, Decision boundary와 상관없는 outlier를 선택하는 경향이 있음.
* 이에 대한 해결책으로 여러 가지 query strategy를 번갈아가면서 사용하는 것임. Exploration-Exploitation Trade-off라고 볼 수 있음.
* 특히나, 데이터에 몇 개의 class가 있는 지 모를 때는 feature space exploration이 더욱 중요해짐. 위에서 언급한 대로 unsupervised clustering 방식을 이용하여 샘플을 선택하는 방법도 권장됨. [How to do Unsupervised Clustering with Keras](https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/)

# Active Learning, part 2: the Practice
* Practice 예제 개요
  * Sandford Dogs Dataset을 이용하여 강아지의 품종을 분류함.
  * Base model로 ImageNet에 학습된 ResNet18에서 마지막 Fully connected layer를 교체하여 fine-tunning하였음.
  * Optimizer는 SGD.
  * 최초 학습시에는 20개이 unlabeled 샘플을 랜덤으로 선택하여 레이블링하고, 그 이후에는 Margin Sampling을 이용하여 레이블링 샘플을 선택함.
  * 모델 학습은 test set에서 Accuracy가 줄어들기 시작하면 종료함. 이외에도 정해진 수의 epoch만 수행하거나, training loss가 임계점 이하로 가면 종료하는 등 여러 stopping criteria가 있을 수 있음. <span style="color:red">(코멘트) 모델 학습에 test set을 사용한 점은 좀 그런데, 실제로 적용할 때는 development set을 따로 확보하는 것이 좋을 것임.<span> 
* Pool size in Pool-based Active Learning
  * Pool-based로 할 때 전체 unlabeled 샘플에서 pool size 만큼의 일부만을 가지고 pool을 구성하면 좋을 때가 있는데, 첫번째로는 unlabeled 데이터셋이 아주 클 때고, 두번째는 아래와 같이 outlier와 관련되어 있음.
  * 데이터에 outlier가 있을 때 부분집합으로 pool을 구성하게 되면 outlier에 대해서 좀 더 강건해짐. 그 이유는 위에서 서술한 대로, Least confidence의 경우 outlier를 선택할 가능성이 높은 상황에서, 예를 들어, 데이터 10개의 outlier가 있다고 할 때, 전체 unlabeled 샘플에서 least confidence 기준으로 20개의 샘플을 선택하면, 이중 10개가 outlier임. 이는 원하지 않은 결과로, 만일 pool size를 전체 unlabeled 샘플의 10%로 설정하면, 해당 pool에는 평균적으로 1개의 oultier만 있을 것이고, 나머지 19개는 informative한 샘플일 것임. <span style="color:red">(코멘트) Pool size 지정은 마치 Random strategy와 Least confidence를 섞어 놓은 형태와 유사한 것으로 보임.</span>
* Results
  * Random sampling과 Margin Sampling의 레이블링된 샘플수에 따른 Test Accuracy 변화를 보면, 
  * 초기에는 Random sampling이 성능이 더 좋게 나옴. 이는 충분한 데이터로 학습하지 못한 모델이 informative한 샘플을 선택하는 데 무작위로 선택하는 것보다 도움이 되지 못한다는 것임.
  * 레이블된 샘플이 추가됨에 따라서, Margin Sampling의 성능이 Random Sampling을 앞서는데, 일부 구간에서 Margin Sampling이 절반 정도의 레이블 데이터를 가지고도 Random Sampling과 비슷한 성능을 내는 것으로 나타남.
  * 레이블된 샘플이 계속 추가됨에 따라서 Margin Sampling과 Random Sampling의 성능차이가 점차 줄어듬.



