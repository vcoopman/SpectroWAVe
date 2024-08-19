#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL_mixer.h>

#include "FileSignalCollector.h"

class DisplayWindow {
  /**
   * User interface for SpectroWAVe.
   * Implemented using SDL2.
   */

  public:
    DisplayWindow(FileSignalCollector* collector);
    ~DisplayWindow();

    /**
     * Checks for SDL events.
     */
    void checkEvent();

    /**
     * Display all draws. 
     */
    void display(std::vector<float> data);

    /**
     * Start a music player in background.
     * Implemented using SDL2/Mixer.
     */
    void startMusic();

  private:
    // Used to retrieve information about the signal.
    FileSignalCollector* collector_;

    const int SCREEN_WIDTH = 1600;
    const int SCREEN_HEIGHT = 800;

    SDL_Window* window_ = NULL;
    SDL_Renderer* renderer_ = NULL;

    TTF_Font* font_;
    const int FONT_SIZE = 24;

    /** Background colors */
    const int BG_RED = 35;
    const int BG_GREEN = 38;
    const int BG_BLUE = 39;
    const int BG_ALPHA = 255;

    /** Bar graph colors */
    const int BARS_BASE_RED = 0;
    const int BARS_BASE_GREEN = 255;
    const int BARS_BASE_BLUE = 0;
    const int BARS_BASE_ALPHA = 255;
    // Values used to computer the final bar color in the bars graph. See setBarColor(...) method.
    const int BARS_COLOR_CHANGE_OVERFLOW_VALUE = 5;
    const int BARS_BASE_COLOR_CHANGE_VALUE = 3;

    /**
     * Cleans window's draws using the background color.
     */
    void cleanWindowDraws();

    /**
     * Used to determine each bar color from the bar graph.
     * It goes from green to purple. Low values will be green, higher values will tend to purple. 
     */
    void setBarColor(float dataPoint, int dataSize);

    /**
     * Draws the bar graph.
     */
    void drawBarsGraph(std::vector<float> data);

    /**
     * Draws text at the specific position using font_.
     */
    void drawText(const std::string& text, int xPosition, int yPosition);

    /**
     * Draws box with information about the tool and the file being played.
     */
    void drawInfoBox();
};
