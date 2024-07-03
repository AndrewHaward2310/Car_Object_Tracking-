#include <Servo.h>

#define UP "UP"
#define DOWN "DOWN"
#define LEFT "LEFT"
#define RIGHT "RIGHT"
#define STOP "STOP"

// Định nghĩa chân cho L298A
const int enA = 4;   // Chân Enable cho động cơ A trên L298A
const int in1 = 34;   // Chân điều khiển 1 cho động cơ A trên L298A
const int in2 = 36;   // Chân điều khiển 2 cho động cơ A trên L298A

const int enB = 5;   // Chân Enable cho động cơ B trên L298A
const int in3 = 30;  // Chân điều khiển 1 cho động cơ B trên L298A
const int in4 = 32;  // Chân điều khiển 2 cho động cơ B trên L298A

void setup() {
  Serial.begin(115200);

  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

}

void moveUp()
{
    analogWrite(enA, 100); // Tốc độ tối đa cho động cơ A trên L298A
    analogWrite(enB, 100); // Tốc độ tối đa cho động cơ B trên L298A
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
}

void moveDown()
{
  analogWrite(enA, 100); // Tốc độ tối đa cho động cơ A trên L298A
  analogWrite(enB, 100); // Tốc độ tối đa cho động cơ B trên L298A
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}
void moveLeft()
{
  analogWrite(enA, 100); // Tốc độ tối đa cho động cơ A trên L298A
  analogWrite(enB, 0); // Tốc độ tối đa cho động cơ B trên L298A
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}

void moveRight()
{
  analogWrite(enA, 0); // Tốc độ tối đa cho động cơ A trên L298A
  analogWrite(enB, 100); // Tốc độ tối đa cho động cơ B trên L298A
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void moveStop()
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}

void moveCar(const char* command) {
  if (strcmp(command, UP) == 0) {
    moveUp();
    delay(200);
    moveStop();
  } else if (strcmp(command, DOWN) == 0) {
    moveDown();
    delay(200);
    moveStop();
  } else if (strcmp(command, RIGHT) == 0) {
    moveRight();
    delay(200);
    moveStop();
  } else if (strcmp(command, LEFT) == 0) {
    moveLeft();
    delay(200);
    moveStop();
    
  } else if (strcmp(command, STOP) == 0) {
    moveStop();
  }
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    Serial.print("Command received: ");
    Serial.println(command);
    
    if (command.startsWith("MOVE")) {
      Serial.println("MOVE command received.");
      moveCar(command.c_str() + 5);
    } 
  }
}
