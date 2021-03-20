let video;
let poseNet;
let pose;
let skeleton;

let numOutputs = 4;

let brain;
let label = "";
let treeAngleTarget = [265.9138,260.6718,272.7146,246.3994,60.34498,111.5485,51.3400,142.6155,116.9140,68.0511,116.1573,59.6353,115.0495,67.7750];
let mountainAngleTarget = [267.9289,267.8780,277.3239,259.9554,67.79409,113.0501,271.2225,274.2705,71.86728,108.5598,71.2390,110.4801,68.2499,114.9507];
let goddessAngleTarget = [225.8889,289.7705,307.4745,237.3843,76.1576,128.9146,153.4326,146.6774,225.0465,128.0132,174.4841,122.9264,227.6440,135.7814];
let warrior2AngleTarget = [254.2574,297.9138,296.2177,242.5748,76.0482,130.3566,257.8660,232.2876,40.0474,134.5878,39.8124,132.0314,82.7860,140.9961];
let sumCalculator = (accumulator, currentValue) => accumulator + currentValue;

let treeAngleTargetSum = treeAngleTarget.reduce(sumCalculator);
let mountainAngleTargetSum = mountainAngleTarget.reduce(sumCalculator);
let goddessAngleTargetSum = goddessAngleTarget.reduce(sumCalculator);
let warrior2AngleTargetSum = warrior2AngleTarget.reduce(sumCalculator);
let tempInputSum;


function setup() {
    createCanvas(640, 480);
    video = createCapture(VIDEO);
    video.hide();
    poseNet = ml5.poseNet(video, modelLoaded);
    poseNet.on('pose', gotPoses);
    // frameRate(30);

    let options = {
        inputs: 14,
        outputs: numOutputs,
        task: 'classification',
        debug: true
    }

    brain = ml5.neuralNetwork(options);
    const modelInfo = {
        model: 'model/model.json',
        metadata: 'model/model_meta.json',
        weights: 'model/model.weights.bin',
    };
    brain.load(modelInfo, brainLoaded);
}

function brainLoaded() {
    console.log('pose classification model ready');
    // if model is ready, then you can begin classification
    classifyPose();
}

function calculate_angle(P1,P2,P3) {
    var angle = (
        Math.atan2(
            P2.position.y - P1.position.y,
            P2.position.x - P1.position.x
        )
        - Math.atan2(
            P3.position.y - P1.position.y,
            P3.position.x - P1.position.x
        )
    ) * (180 / Math.PI);
    if (angle > 90) {
        angle = 450 - angle;
    } else {
        angle = 90 - angle;
    }
    return angle; //Rounds the angle to 4 decimal places
}

function classifyPose(){
    // only classify if there is a pose detected from posenet
    if(pose){
        let inputs = [];
        let tempInput = [];
        // angle is denoted by angle(P1,P2,P3) where P1 is the 'origin'
        let lKnee_lAnkle_lHip = calculate_angle(pose.keypoints[13], pose.keypoints[15], pose.keypoints[11]);
        let rKnee_rAnkle_rHip = calculate_angle(pose.keypoints[14], pose.keypoints[16], pose.keypoints[12]);
        let a0 = inputs.push(lKnee_lAnkle_lHip);
        let a1 = inputs.push(rKnee_rAnkle_rHip);
        tempInput.push(lKnee_lAnkle_lHip);
        tempInput.push(rKnee_rAnkle_rHip);


        let lHip_lKnee_lShoulder = calculate_angle(pose.keypoints[11], pose.keypoints[13], pose.keypoints[5]);
        let rHip_rKnee_rShoulder = calculate_angle(pose.keypoints[12], pose.keypoints[14], pose.keypoints[6]);
        let a2 = inputs.push(lHip_lKnee_lShoulder);
        let a3 = inputs.push(rHip_rKnee_rShoulder);
        tempInput.push(lHip_lKnee_lShoulder);
        tempInput.push(rHip_rKnee_rShoulder);

        let lShoulder_lHip_lElbow = calculate_angle(pose.keypoints[5], pose.keypoints[11], pose.keypoints[7]);
        let rShoulder_rHip_rElbow = calculate_angle(pose.keypoints[6], pose.keypoints[12], pose.keypoints[8]);
        let a4 = inputs.push(lShoulder_lHip_lElbow);
        let a5 = inputs.push(rShoulder_rHip_rElbow);
        tempInput.push(lShoulder_lHip_lElbow);
        tempInput.push(rShoulder_rHip_rElbow);

        let lElbow_lShoulder_lWrist = calculate_angle(pose.keypoints[7], pose.keypoints[5], pose.keypoints[9]);
        let rElbow_rShoulder_rWrist = calculate_angle(pose.keypoints[8], pose.keypoints[6], pose.keypoints[10]);
        let a6 = inputs.push(lElbow_lShoulder_lWrist);
        let a7 = inputs.push(rElbow_rShoulder_rWrist);
        tempInput.push(lElbow_lShoulder_lWrist);
        tempInput.push(rElbow_rShoulder_rWrist);


        let lShoulder_lAnkle_lWrist = calculate_angle(pose.keypoints[5], pose.keypoints[15], pose.keypoints[9]);
        let rShoulder_rAnkle_rWrist = calculate_angle(pose.keypoints[6], pose.keypoints[16], pose.keypoints[10]);
        let a8 = inputs.push(lShoulder_lAnkle_lWrist);
        let a9 = inputs.push(rShoulder_rAnkle_rWrist);
        tempInput.push(lShoulder_lAnkle_lWrist);
        tempInput.push(rShoulder_rAnkle_rWrist);

        let lShoulder_lKnee_lWrist = calculate_angle(pose.keypoints[5], pose.keypoints[13], pose.keypoints[9]);
        let rShoulder_rKnee_rWrist = calculate_angle(pose.keypoints[6], pose.keypoints[14], pose.keypoints[10]);
        let a10 = inputs.push(lShoulder_lKnee_lWrist);
        let a11 = inputs.push(rShoulder_rKnee_rWrist);
        tempInput.push(lShoulder_lKnee_lWrist);
        tempInput.push(rShoulder_rKnee_rWrist);

        let lShoulder_lHip_lWrist = calculate_angle(pose.keypoints[5], pose.keypoints[11], pose.keypoints[9]);
        let rShoulder_rHip_rWrist = calculate_angle(pose.keypoints[6], pose.keypoints[12], pose.keypoints[10]);
        let a12 = inputs.push(lShoulder_lHip_lWrist);
        let a13 = inputs.push(rShoulder_rHip_rWrist);
        tempInput.push(lShoulder_lHip_lWrist);
        tempInput.push(rShoulder_rHip_rWrist);


        brain.classify(inputs, gotResult);//Gets the points to classify
        /*tempInput.push(a0);
        tempInput.push(a1);
        tempInput.push(a2);
        tempInput.push(a3);
        tempInput.push(a4);
        tempInput.push(a5);
        tempInput.push(a6);
        tempInput.push(a7);
        tempInput.push(a8);
        tempInput.push(a9);
        tempInput.push(a10);
        tempInput.push(a11);
        tempInput.push(a12);
        tempInput.push(a13);*/
        tempInputSum = tempInput.reduce(sumCalculator);
        //console.log(tempInputSum);
        verifyTreePose(tempInput);

    }
    else{
        // delay if it wasnt able to detect the initial pose
        setTimeout(classifyPose, 100);
    }
}

function gotResult(error, results){
    if(results[0].confidence > 0.75){
        label = results[0].label;
        console.log(label);
    }
    // after first classification, you want to keep classifying for new poses
    classifyPose();
}

function verifyTreePose(calculatedAngle){
  let x;
  let y;

  for(i = 0; i<treeAngleTarget.length; i++){
   if (calculatedAngle[i] < treeAngleTarget[i] + 10.0 && calculatedAngle[i] > treeAngleTarget[i] - 10.0){
    //For Vatsal + Dimple of specific angles relating to pose
    //green light + angles are correct + feedback
    }
    else if (i===0){
      text('Fix your left knee + ankle + hip',350,200);
      pose.keypoints[13].position.x;
      pose.keypoints[13].position.y;
      pose.keypoints[15].position.x;
      pose.keypoints[15].position.y;
      pose.keypoints[11].position.x;
      pose.keypoints[11].position.y;
      fill (255, 51, 0);
      ellipse(x, y, 16, 16);
    }
    else if (i===1){
      text('Fix your right knee + ankle + hip',350,200);
      pose.keypoints[14].position.x;
      pose.keypoints[14].position.y;
      pose.keypoints[16].position.x;
      pose.keypoints[16].position.y;
      pose.keypoints[12].position.x;
      pose.keypoints[12].position.y;
      fill (255, 51, 0);
      ellipse(x, y, 16, 16);
    }
    else if (i===2){
      text('Fix your left hip + knee + shoulder',350,200);
    }
    else if (i===3){
      text('Fix your right hip + knee + shoulder',350,200);
    }
    else if (i===4){
      text('Fix your left shoulder + hip + elbow',350,200);
    }
    else if (i===5){
      text('Fix your right shoulder + hip + elbow',350,200);
    }
    else if (i===6){
      text('Fix your left elbow + shoulder + wrist',350,200);
    }
    else if (i===7){
      text('Fix your right elbow + shoulder + wrist',350,200);
    }
    else if (i===8){
      text('Fix your left shoulder + ankle + wrist',350,200);
    }
    else if (i===9){
      text('Fix your right shoulder + ankle + wrist',350,200);
    }
    else if (i===10){
      text('Fix your left shoulder + knee + wrist',350,200);
    }
    else if (i===11){
      text('Fix your right shoulder + knee + wrist',350,200);
    }
    else if (i===12){
      text('Fix your left shoulder + hip + wrist',350,200);
    }
    else if (i===13){
      text('Fix your right shoulder + hip + wrist',350,200);
    }





      //switch block


  }
/*Overall Percent accuracy for Tree Pose*/
  if(treeAngleTargetSum>=tempInputSum){
    let treeAngleAccuracy1 = ((treeAngleTargetSum - tempInputSum)/treeAngleTargetSum)*100;
    text('Correct angle', treeAngleAccuracy1);
  }
  else if(treeAngleTargetSum<tempInputSum) {
    let treeAngleAccuracy2 = ((tempInputSum-treeAngleTargetSum)/treeAngleTargetSum)*100;
    text('Correct angle', treeAngleAccuracy2);
  }
}

function verifyMountainPose(calculatedAngle){
  for(i = 0; i<mountainAngleTarget.length; i++){
    if (calculatedAngle[i] < mountainAngleTarget[i] + 10.0 && calculatedAngle[i] > mountainAngleTarget[i] - 10.0){
    //For Vatsal + Dimple of specific angles relating to pose
    }
    else {
      text('Fix your angle');
    }
  }
/*Overall Percent accuracy for Mountain Pose*/
  if(mountainAngleTargetSum>=tempInputSum){
    let mountainAngleAccuracy1 = ((mountainAngleTargetSum - tempInputSum)/mountainAngleTargetSum)*100;
    text('Correct angle', mountainAngleAccuracy1);
  }
  else if(mountainAngleTargetSum<tempInputSum) {
    let mountainAngleAccuracy2 = ((tempInputSum-mountainAngleTargetSum)/mountainAngleTargetSum)*100;
    text('Correct angle', mountainAngleAccuracy2);
  }
}

function verifyGoddessPose(calculatedAngle){
  for(i = 0; i<goddessAngleTarget.length; i++){
    if (calculatedAngle[i] < goddessAngleTarget[i] + 10.0 && calculatedAngle[i] > goddessAngleTarget[i] - 10.0){
    //For Vatsal + Dimple of specific angles relating to pose
    }
    else {
      text('Fix your angle');
    }
  }
/*Overall Percent accuracy for Tree Pose*/
  if(goddessAngleTargetSum>=tempInputSum){
    let goddessAngleAccuracy1 = ((goddessAngleTargetSum - tempInputSum)/goddessAngleTargetSum)*100;
    text('Correct angle', goddessAngleAccuracy1);
  }
  else if(goddessAngleTargetSum<tempInputSum) {
    let goddessAngleAccuracy2 = ((tempInputSum-goddessAngleTargetSum)/goddessAngleTargetSum)*100;
    text('Correct angle', goddessAngleAccuracy2);
  }
}

function verifyWarrior2Pose(calculatedAngle){
  for(i = 0; i<warrior2AngleTarget.length; i++){
    if (calculatedAngle[i] < warrior2AngleTarget[i] + 10.0 && calculatedAngle[i] > warrior2AngleTarget[i] - 10.0){
    //For Vatsal + Dimple of specific angles relating to pose
    }
    else {
      text('Fix your angle');
    }
  }
/*Overall Percent accuracy for Tree Pose*/
  if(warrior2AngleTargetSum>=tempInputSum){
    let warrior2AngleAccuracy1 = ((warrior2AngleTargetSum - tempInputSum)/warrior2AngleTargetSum)*100;
    text('Correct angle', warrior2AngleAccuracy1);
  }
  else if(warrior2AngleTargetSum<tempInputSum) {
    let warrior2AngleAccuracy2 = ((tempInputSum-warrior2AngleTargetSum)/warrior2AngleTargetSum)*100;
    text('Correct angle', warrior2AngleAccuracy2);
  }
}



function gotPoses(poses) {
    console.log(poses);
    if (poses.length > 0) {
        pose = poses[0].pose;
        skeleton = poses[0].skeleton;
    }
}

function modelLoaded () {
    console.log('poseNet ready');
}

function draw() {
    push();
    translate(video.width,0);
    scale(-1,1);
    image(video, 0, 0, video.width, video.height);
  /*  if (pose) {
        for (let i = 0; i < pose.keypoints.length; i++){
            let x = pose.keypoints[i].position.x;
            let y = pose.keypoints[i].position.y;
            fill (100, 99, 82);//fill(255, 51, 0);
            ellipse(x, y, 16, 16);
        }

        for (let i = 0; i < skeleton.length; i++){
            let a = skeleton[i][0]; //skeleton is a 2D array, the second dimension holds the 2 locations that are connected on the keypoint
            let b = skeleton[i][1];
            strokeWeight(2);
            stroke(255);
            line(a.position.x, a.position.y, b.position.x, b.position.y);
        }*/

        /*pose.keypoints[5].position.x;
        pose.keypoints[5].position.y;
        fill (255, 51, 0);
        ellipse(x, y, 16, 16);*/
    //}
    pop();
    fill(255);
    textSize(128);
    textAlign(CENTER,TOP);
    text(label, 0, 12, width);
    // text("FRate "+ int(getFrameRate()),350,200);
}
