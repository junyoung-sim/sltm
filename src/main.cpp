
#include <iostream>
#include "../lib/encoder.hpp"

int main(int argc, char *argv[])
{
    Encoder model(argv[1]); // argv[1] = model name
    model.add_layer(2, 1, false, "max", 2); // *** HARD-CODED PARAMETER ***
    model.load();
    model.encode();
    model.save();
    return 0;
}
