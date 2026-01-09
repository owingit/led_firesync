#ifndef FireflyLED_h
#define FireflyLED_h
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

#endif
