#include "Arduino.h"

class FireflyLED
{
    public:
        FireflyLED(int pin, unsigned long interval_on, unsigned long interval_off, int number_of_flashes);
        void update_interval_wait();
        int get_flashing_state();
        void increment_charge();
        void check_and_write(unsigned long currentMillis);


    private:
        unsigned long interval_wait;
        unsigned long interval_on, interval_off, interval_off_always, previousMillis;
        int is_flashing;
        int flash_counter;
        int pin_number;
        int charge;
        int state;
        int charging_threshold;
        int number_of_flashes;
        int brightness;
        
};

int get_number_of_flashes();
unsigned long get_interval_on();
unsigned long get_interval_off();

//from data
unsigned long get_interval_on()
{   
    return 10;
}

//from data
unsigned long get_interval_off()
{
    return 770; //850; 700; etc
      
}

//from data
int get_number_of_flashes()
{   
    int retval = 20;
    return retval;
}

FireflyLED::FireflyLED(int pin, unsigned long interval_on, unsigned long interval_off, int number_of_flashes)
{
    pinMode(pin, OUTPUT);
    this->pin_number = pin;
    this->interval_on = interval_on;
    this->interval_off_always = interval_off;
    this->interval_off = 65000; // sleep for 1.5min when plugged in
    this->interval_wait = 100; //random(12000, 24000) - ((interval_on + interval_off) * number_of_flashes);
    this->previousMillis = 0;
    this->state = LOW;
    this->charge = 0;
    this->charging_threshold = (interval_on + interval_off) * number_of_flashes; // refractory period. should this relate to the interflash-interval? or the female cadence?
    this->brightness = 1; //0-255 //MEGA only
    this->number_of_flashes = number_of_flashes;
    this->flash_counter = number_of_flashes;
    this->is_flashing = 0;
}

void FireflyLED::update_interval_wait() 
{   
    if (this->charge > this->charging_threshold) {
        this->interval_off = this->interval_off_always;
    }
}

int FireflyLED::get_flashing_state()
{
    return this->is_flashing;
}

void FireflyLED::increment_charge()
{
    this->charge += 1;
}

void FireflyLED::check_and_write(unsigned long currentMillis) 
{ 
    if (this->state == LOW) { // if is off, wait interval_off
        if (currentMillis - this->previousMillis >= this->interval_off) {
            // save the last time you blinked the LED
            this->previousMillis = currentMillis;
            // if the LED is off turn it on and vice-versa:
            this->state = HIGH;
            this->is_flashing = 1;
            this->charge = 0;
            //analogWrite(this->pin_number, this->brightness); //MEGA only
            digitalWrite(this->pin_number, this->state);
        }
    }
    else { // else wait interval_on
        if (currentMillis - this->previousMillis >= this->interval_on) {
            // save the last time you blinked the LED
            this->previousMillis = currentMillis;
            // if the LED is off turn it on and vice-versa:
            this->state = LOW;
            this->is_flashing = 0;
            this->flash_counter -= 1;
            if (this->flash_counter == 0) {
                this->state = LOW;
//                this->interval_off = this->interval_wait;
                this->flash_counter = this->number_of_flashes;
//                get_number_of_flashes();
//                this->number_of_flashes = this->flash_counter;
            }
            else if (this->flash_counter <= this->number_of_flashes - 1) {
                this->interval_off = this->interval_off_always;
            }
            //analogWrite(this->pin_number, 0); //MEGA only
            digitalWrite(this->pin_number, this->state);
        }
        
    }
}

//instantiate FireflyLEDs
FireflyLED ff1(3, get_interval_on(), get_interval_off(), get_number_of_flashes());
FireflyLED myLEDs[1]= {ff1};


void setup() {
    Serial.begin(9600);
}

void loop() {
    int count_flashing = 0;
    unsigned long currentMillis = millis();
    
    for (byte i = 0; i < sizeof(myLEDs) / sizeof(myLEDs[0]); i = i+1) {
        myLEDs[i].check_and_write(currentMillis);
//        myLEDs[i].increment_charge();
//        if (myLEDs[i].get_flashing_state() == 1) {
//            for (byte j = 0; j < sizeof(myLEDs) / sizeof(myLEDs[0]); j = j+1) {
//                if (j != i) {
//                    myLEDs[j].update_interval_wait();
//                }
//            }
//        }
//        count_flashing += myLEDs[i].get_flashing_state();
    }
}
