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
    DisplayWindow(std::string filepath, int sampleRate, int channels, int frames, int fftSize, int binning);
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
    const int SCREEN_WIDTH = 1600;
    const int SCREEN_HEIGHT = 400;

    SDL_Window* window_ = NULL;
    SDL_Renderer* renderer_ = NULL;

    TTF_Font* font_;
    const int FONT_SIZE = 18;

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

    /** InfoBox colors */
    const int INFO_BOX_BG_RED = 0;
    const int INFO_BOX_BG_GREEN = 0;
    const int INFO_BOX_BG_BLUE = 0;
    const int INFO_BOX_BG_ALPHA = 0;
    const int INFO_BOX_BORDER_RED = 255;
    const int INFO_BOX_BORDER_GREEN = 255;
    const int INFO_BOX_BORDER_BLUE = 255;
    const int INFO_BOX_BORDER_ALPHA = 255;
    const Uint8 INFO_BOX_TEXT_RED = 255;
    const Uint8 INFO_BOX_TEXT_GREEN = 255;
    const Uint8 INFO_BOX_TEXT_BLUE = 255;
    const Uint8 INFO_BOX_TEXT_ALPHA = 255;

    /** Information about the signal being displayed */
    std::string filepath_;
    int sampleRate_;
    int channels_;
    int frames_;
    int fftSize_;
    int binning_;

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
    void drawText(const std::string& text, int xPosition, int yPosition, SDL_Color textColor);

    /**
     * Draws box with information about the tool and the file being played.
     */
    void drawInfoBox();
};
