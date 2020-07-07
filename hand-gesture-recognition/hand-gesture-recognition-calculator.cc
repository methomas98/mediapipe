#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

// Strings
//#include <stdlib.h>
//#include <bits/stdc++.h>

// Named pipe
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace mediapipe
{

namespace
{
constexpr char normRectTag[] = "NORM_RECT";
constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
constexpr char recognizedHandGestureTag[] = "RECOGNIZED_HAND_GESTURE";
} // namespace

// Graph config:
//
// node {
//   calculator: "HandGestureRecognitionCalculator"
//   input_stream: "NORM_LANDMARKS:scaled_landmarks"
//   input_stream: "NORM_RECT:hand_rect_for_next_frame"
// }
class HandGestureRecognitionCalculator : public CalculatorBase
{
public:
    static ::mediapipe::Status GetContract(CalculatorContract *cc);
    ::mediapipe::Status Open(CalculatorContext *cc) override;

    ::mediapipe::Status Process(CalculatorContext *cc) override;

    // Named pipe
    int fd;
    char * myfifo = "/tmp/myfifo";

private:
    float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x, float b_y)
    {
        float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
        return std::sqrt(dist);
    }

    bool isThumbNearFirstFinger(NormalizedLandmark point1, NormalizedLandmark point2)
    {
        float distance = this->get_Euclidean_DistanceAB(point1.x(), point1.y(), point2.x(), point2.y());
        return distance < 0.1;
    }
};

REGISTER_CALCULATOR(HandGestureRecognitionCalculator);

::mediapipe::Status HandGestureRecognitionCalculator::GetContract(
    CalculatorContract *cc)
{
    RET_CHECK(cc->Inputs().HasTag(normalizedLandmarkListTag));
    cc->Inputs().Tag(normalizedLandmarkListTag).Set<mediapipe::NormalizedLandmarkList>();

    RET_CHECK(cc->Inputs().HasTag(normRectTag));
    cc->Inputs().Tag(normRectTag).Set<NormalizedRect>();

    RET_CHECK(cc->Outputs().HasTag(recognizedHandGestureTag));
    cc->Outputs().Tag(recognizedHandGestureTag).Set<std::string>();

    return ::mediapipe::OkStatus();
}

::mediapipe::Status HandGestureRecognitionCalculator::Open(
    CalculatorContext *cc)
{
    cc->SetOffset(TimestampDiff(0));

    // NAMED PIPE
    mkfifo(myfifo, 0666);

    return ::mediapipe::OkStatus();
}

::mediapipe::Status HandGestureRecognitionCalculator::Process(
    CalculatorContext *cc)
{
    std::string *recognized_hand_gesture;

    // hand closed (red) rectangle
    const auto rect = &(cc->Inputs().Tag(normRectTag).Get<NormalizedRect>());
    float width = rect->width();
    float height = rect->height();

    if (width < 0.01 || height < 0.01)
    {
        // LOG(INFO) << "No Hand Detected";
        recognized_hand_gesture = new std::string("___");

        cc->Outputs()
            .Tag(recognizedHandGestureTag)
            .Add(recognized_hand_gesture, cc->InputTimestamp());
        return ::mediapipe::OkStatus();
    }

    const auto &landmarkList = cc->Inputs()
                                   .Tag(normalizedLandmarkListTag)
                                   .Get<mediapipe::NormalizedLandmarkList>();
    RET_CHECK_GT(landmarkList.landmark_size(), 0) << "Input landmark vector is empty.";

    // finger states
    bool thumbIsOpen = false;
    bool firstFingerIsOpen = false;
    bool secondFingerIsOpen = false;
    bool thirdFingerIsOpen = false;
    bool fourthFingerIsOpen = false;
    //

    // My edits: June 30
    // Orientation: true if the thumb is on the left side of the "normalized" image
    bool orientation = true;
    
    if (landmarkList.landmark(5).x() > landmarkList.landmark(17).x()){
        orientation = false;
    }
    //LOG(INFO) << "Orientation: " << orientation;
    
    // Hand is oriented with thumb on left side (orientation == True)
    float pseudoFixKeyPoint = landmarkList.landmark(2).x();
    if (orientation && landmarkList.landmark(3).x() < pseudoFixKeyPoint && landmarkList.landmark(4).x() < pseudoFixKeyPoint)
    {
        thumbIsOpen = true;
    }

    // Hand is oriented with thumb on right side (orientation == False)
    if (!orientation && landmarkList.landmark(3).x() > pseudoFixKeyPoint && landmarkList.landmark(4).x() > pseudoFixKeyPoint)
    {
        thumbIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(6).y();
    //if (landmarkList.landmark(7).y() < pseudoFixKeyPoint && landmarkList.landmark(8).y() < pseudoFixKeyPoint)
    if (landmarkList.landmark(7).y() < pseudoFixKeyPoint && landmarkList.landmark(8).y() < landmarkList.landmark(7).y())
    {
        firstFingerIsOpen = true;

        // But if the tip of the finger is close to the base, finger should be closed (account for errors when close together)
        if(this->isThumbNearFirstFinger(landmarkList.landmark(5), landmarkList.landmark(8))){
            firstFingerIsOpen = false;
        }

    }

    pseudoFixKeyPoint = landmarkList.landmark(10).y();
    //if (landmarkList.landmark(11).y() < pseudoFixKeyPoint && landmarkList.landmark(12).y() < pseudoFixKeyPoint)
    if (landmarkList.landmark(11).y() < pseudoFixKeyPoint && landmarkList.landmark(12).y() < landmarkList.landmark(11).y())
    {
        secondFingerIsOpen = true;

         // But if the tip of the finger is close to the base, finger should be closed (account for errors when close together)
        if(this->isThumbNearFirstFinger(landmarkList.landmark(9), landmarkList.landmark(12))){
            secondFingerIsOpen = false;
        }

   }

    pseudoFixKeyPoint = landmarkList.landmark(14).y();
    //if (landmarkList.landmark(15).y() < pseudoFixKeyPoint && landmarkList.landmark(16).y() < pseudoFixKeyPoint)
    if (landmarkList.landmark(15).y() < pseudoFixKeyPoint && landmarkList.landmark(16).y() < landmarkList.landmark(15).y())
    {
        thirdFingerIsOpen = true;

        // But if the tip of the finger is close to the base, finger should be closed (account for errors when close together)
        if(this->isThumbNearFirstFinger(landmarkList.landmark(13), landmarkList.landmark(16))){
            thirdFingerIsOpen = false;
        }

    }

    pseudoFixKeyPoint = landmarkList.landmark(18).y();
    //if (landmarkList.landmark(19).y() < pseudoFixKeyPoint && landmarkList.landmark(20).y() < pseudoFixKeyPoint)
    if (landmarkList.landmark(19).y() < pseudoFixKeyPoint && landmarkList.landmark(20).y() < landmarkList.landmark(19).y())
    {
        fourthFingerIsOpen = true;

        // But if the tip of the finger is close to the base, finger should be closed (account for errors when close together)
        if(this->isThumbNearFirstFinger(landmarkList.landmark(17), landmarkList.landmark(20))){
            fourthFingerIsOpen = false;
        }

    }

    // Create 8 bit integer with the bit "finger states"
    u_int8_t States = 0x00;
    States = (fourthFingerIsOpen << 4) | (thirdFingerIsOpen << 3) | (secondFingerIsOpen << 2) | (firstFingerIsOpen << 1) |  (thumbIsOpen << 0); 

    // Hand gesture recognition
    switch(States)
    {
        case 0b11111:
        recognized_hand_gesture = new std::string("FIVE");
        break;

        case 0b11110:
        recognized_hand_gesture = new std::string("FOUR");
        break;

        case 0b00111:
        recognized_hand_gesture = new std::string("THREE");
        break;

        case 0b01110:
        recognized_hand_gesture = new std::string("THREE");
        break;

        case 0b00011:
        recognized_hand_gesture = new std::string("CHECK");
        break;

        case 0b00110:
        recognized_hand_gesture = new std::string("TWO");
        break;

        case 0b00010:
        recognized_hand_gesture = new std::string("ONE");
        break;

        case 0b10010:
        recognized_hand_gesture = new std::string("ROCK");
        break;

        case 0b10011:
        recognized_hand_gesture = new std::string("SPIDERMAN");
        break;

        case 0b00001:
        recognized_hand_gesture = new std::string("THUMBS UP"); 
        break;

        case 0b00100:
        recognized_hand_gesture = new std::string("EXPLETIVE"); 
        break;

        case 0b00000:
        recognized_hand_gesture = new std::string("FIST"); 
        break;

        case 0b10000:
        recognized_hand_gesture = new std::string("PINKY"); 
        break;

        case 0b10001:
        recognized_hand_gesture = new std::string("HANG LOOSE"); 
        break;

        default:
        recognized_hand_gesture = new std::string("___");
        //LOG(INFO) << "Finger States: " << thumbIsOpen << firstFingerIsOpen << secondFingerIsOpen << thirdFingerIsOpen << fourthFingerIsOpen;       
        break;

    }

    // This case will overwrite case "FIVE"
    //if (!firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen && this->isThumbNearFirstFinger(landmarkList.landmark(4), landmarkList.landmark(8)))
    if ( ((States|0x01) == 0b11101)  && this->isThumbNearFirstFinger(landmarkList.landmark(4), landmarkList.landmark(8)))
    {
        recognized_hand_gesture = new std::string("PERFECT");
    }

    // Named pipe
    const char * message = (*recognized_hand_gesture).c_str();
    fd = open(myfifo, O_WRONLY);
    write(fd,message,strlen(message)+1); 
    close(fd);

    // LOG(INFO) << recognized_hand_gesture;
    //LOG(INFO) << "Finger States: " << thumbIsOpen << firstFingerIsOpen << secondFingerIsOpen << thirdFingerIsOpen << fourthFingerIsOpen;       

    cc->Outputs()
        .Tag(recognizedHandGestureTag)
        .Add(recognized_hand_gesture, cc->InputTimestamp());

    return ::mediapipe::OkStatus();
} // namespace mediapipe

} // namespace mediapipe
