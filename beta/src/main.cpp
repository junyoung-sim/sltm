
#include <iostream>
#include "../lib/encoder.hpp"

int main(int argc, char *argv[])
{
    Encoder model("test");
    model.add_layer(2, 1, false, "max", 2);
    model.load();
    model.encode();
    model.save();
    return 0;
}
