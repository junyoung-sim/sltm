
#include <iostream>
#include "../lib/futures.hpp"

int main()
{
    Futures model("test");
    model.add_layer(2, 1, false, "max", 2);
    model.load();
    model.save();
    return 0;
}
