   // 引入 TensorFlow.js 和 BlazePose 相關庫
    const tf = window.tf;
    const poseDetection = window.poseDetection;

    // 加載 BlazePose 模型
    async function loadBlazePoseModel() {
      const model = await poseDetection.createDetector(poseDetection.SupportedModels.BlazePose);
      return model;
    }

    // 進行人體姿勢估計
    async function estimatePose(model, videoElement) {
      const predictions = await model.estimatePoses(videoElement);
      return predictions;
    }

    // 範例用法
    async function runExample() {
      // 加載 BlazePose 模型
      const model = await loadBlazePoseModel();

      // 捕獲視訊流
      const videoElement = document.getElementById('video');
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoElement.srcObject = stream;

      // 等待視訊加載完成
      await new Promise((resolve) => {
        videoElement.onloadedmetadata = resolve;
      });

      // 進行人體姿勢估計
      const predictions = await estimatePose(model, videoElement);

      // 繪製預測結果
      const canvasElement = document.getElementById('canvas');
      const context = canvasElement.getContext('2d');
      predictions.forEach((pose) => {
        pose.keypoints.forEach((keypoint) => {
          const { x, y } = keypoint.position;
          context.beginPath();
          context.arc(x, y, 5, 0, 2 * Math.PI);
          context.fillStyle = 'red';
          context.fill();
        });
      });
    }

    // 執行示例
    runExample();
