let video;
let poseNet;
let pose;
let skeleton;

let numOutputs = 4;

let brain;
let label = "";

function setup() {
    var canvas = createCanvas(800, 600);
    canvas.position(375, 50);
    video = createCapture(VIDEO);
    video.hide();
    video.size(800, 600);
    poseNet = ml5.poseNet(video, modelLoaded);
    poseNet.on('pose', gotPoses);

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
    return angle;
}

function classifyPose(){
    // only classify if there is a pose detected from posenet
    if(pose){
        let inputs = [];
        // angle is denoted by angle(P1,P2,P3) where P1 is the 'origin'
        let lKnee_lAnkle_lHip = calculate_angle(pose.keypoints[13], pose.keypoints[15], pose.keypoints[11]);
        let rKnee_rAnkle_rHip = calculate_angle(pose.keypoints[14], pose.keypoints[16], pose.keypoints[12]);
        inputs.push(lKnee_lAnkle_lHip);
        inputs.push(rKnee_rAnkle_rHip);

        let lHip_lKnee_lShoulder = calculate_angle(pose.keypoints[11], pose.keypoints[13], pose.keypoints[5]);
        let rHip_rKnee_rShoulder = calculate_angle(pose.keypoints[12], pose.keypoints[14], pose.keypoints[6]);
        inputs.push(lHip_lKnee_lShoulder);
        inputs.push(rHip_rKnee_rShoulder);

        let lShoulder_lHip_lElbow = calculate_angle(pose.keypoints[5], pose.keypoints[11], pose.keypoints[7]);
        let rShoulder_rHip_rElbow = calculate_angle(pose.keypoints[6], pose.keypoints[12], pose.keypoints[8]);
        inputs.push(lShoulder_lHip_lElbow);
        inputs.push(rShoulder_rHip_rElbow);

        let lElbow_lShoulder_lWrist = calculate_angle(pose.keypoints[7], pose.keypoints[5], pose.keypoints[9]);
        let rElbow_rShoulder_rWrist = calculate_angle(pose.keypoints[8], pose.keypoints[6], pose.keypoints[10]);
        inputs.push(lElbow_lShoulder_lWrist);
        inputs.push(rElbow_rShoulder_rWrist);

        let lShoulder_lAnkle_lWrist = calculate_angle(pose.keypoints[5], pose.keypoints[15], pose.keypoints[9]);
        let rShoulder_rAnkle_rWrist = calculate_angle(pose.keypoints[6], pose.keypoints[16], pose.keypoints[10]);
        inputs.push(lShoulder_lAnkle_lWrist);
        inputs.push(rShoulder_rAnkle_rWrist);

        let lShoulder_lKnee_lWrist = calculate_angle(pose.keypoints[5], pose.keypoints[13], pose.keypoints[9]);
        let rShoulder_rKnee_rWrist = calculate_angle(pose.keypoints[6], pose.keypoints[14], pose.keypoints[10]);
        inputs.push(lShoulder_lKnee_lWrist);
        inputs.push(rShoulder_rKnee_rWrist);

        let lShoulder_lHip_lWrist = calculate_angle(pose.keypoints[5], pose.keypoints[11], pose.keypoints[9]);
        let rShoulder_rHip_rWrist = calculate_angle(pose.keypoints[6], pose.keypoints[12], pose.keypoints[10]);
        inputs.push(lShoulder_lHip_lWrist);
        inputs.push(rShoulder_rHip_rWrist);

        brain.classify(inputs, gotResult);
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
    if (pose) {
        for (let i = 0; i < pose.keypoints.length; i++){
            let x = pose.keypoints[i].position.x;
            let y = pose.keypoints[i].position.y;
            fill (100, 99, 82);
            ellipse(x, y, 16, 16);
        }

        for (let i = 0; i < skeleton.length; i++){
            let a = skeleton[i][0]; //skeleton is a 2D array, the second dimension holds the 2 locations that are connected on the keypoint
            let b = skeleton[i][1];
            strokeWeight(2);
            stroke(255);
            line(a.position.x, a.position.y, b.position.x, b.position.y);
        }
    }
    pop();
    fill(0, 76, 154);
    textSize(100);
    textAlign(CENTER,TOP);
    text(label, 0, 12, width);
}
