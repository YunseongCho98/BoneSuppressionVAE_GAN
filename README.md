# Bone Suppression using VAE-GAN

이 프로젝트는 Variational Autoencoder(VAE)를 생성자로 활용하고, GAN(Generative Adversarial Network) 구조를 결합하여 Bone Suppression을 수행하는 모델입니다. VAE는 입력 이미지를 잠재 공간(latent space)으로 인코딩하고, 이를 다시 디코딩하여 뼈 구조가 억제된 이미지를 생성합니다. GAN의 판별자는 생성된 이미지와 실제 이미지를 구분하며, 이를 통해 생성자의 성능을 향상시킵니다.

---

## 파라미터 다운로드

- [파라미터 다운로드 링크](https://drive.google.com/file/d/17CQJCCyzVfMgBPv_rDdjIkcbywIEMywB/view?usp=share_link)

## 학습 및 테스트 데이터 다운로드

- [데이터 다운로드 링크](https://drive.google.com/file/d/1VKbe_xsSXblG1v7SKj9Fw0n7gNE2KP0E/view?usp=share_link)