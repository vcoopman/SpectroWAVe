#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

class DisplayWindow {
  public:
    DisplayWindow();
    ~DisplayWindow();

    void checkEvent();
    void display(std::vector<float> data);

  private:
    const int SCREEN_WIDTH = 1600;
    const int SCREEN_HEIGHT = 800;

    // Base RGB matches Winamp Green.
    const int BASE_RED = 34;
    const int BASE_GREEN = 161;
    const int BASE_BLUE = 37;
    const int BASE_ALPHA = 255;

    const int COLOR_CHANGE_OVERFLOW_VALUE = 100;
    const int BASE_COLOR_CHANGE_VALUE = 3;

    SDL_Window* window_ = NULL;
    SDL_Renderer* renderer_ = NULL;

    TTF_Font* font_;
    const int FONT_SIZE = 10;

    void cleanWindowDraws();
    void setBarColor(float dataPoint, int dataSize);
    void drawBarsGraph(std::vector<float> data);
    void drawText(const std::string& text, int xPosition, int yPosition);
};
