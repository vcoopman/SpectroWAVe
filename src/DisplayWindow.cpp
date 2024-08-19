#include "DisplayWindow.h"


DisplayWindow::DisplayWindow(std::string filepath, int sampleRate, int channels, int frames, int fftSize, int binning) :
  filepath_(filepath),
  sampleRate_(sampleRate),
  channels_(channels),
  frames_(frames),
  fftSize_(fftSize),
  binning_(binning)
{
  if (SDL_Init( SDL_INIT_VIDEO ) < 0)
    throw std::runtime_error("SDL could not initialize! SDL_Error: " + std::string(SDL_GetError())); 

  window_ = SDL_CreateWindow(
    "~SpectroWAVe 🌈⃤",
    SDL_WINDOWPOS_UNDEFINED,
    SDL_WINDOWPOS_UNDEFINED,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SDL_WINDOW_SHOWN
  );

  if(window_ == NULL)
    throw std::runtime_error("Window could not be created! SDL_Error: " + std::string(SDL_GetError())); 

  // Disable specific events.
  // This is done to reduce the amount of events produced by SDL
  // and avoid looping through unwanted events.
  SDL_EventState(SDL_MOUSEMOTION, SDL_IGNORE);
  SDL_EventState(SDL_MOUSEBUTTONDOWN, SDL_IGNORE);
  SDL_EventState(SDL_MOUSEBUTTONUP, SDL_IGNORE);
  SDL_EventState(SDL_MOUSEWHEEL, SDL_IGNORE);
  SDL_EventState(SDL_WINDOWEVENT, SDL_IGNORE);
  SDL_EventState(SDL_SYSWMEVENT, SDL_IGNORE);       

  renderer_ = SDL_CreateRenderer(
    window_,
    -1,
    SDL_RENDERER_ACCELERATED
  );

  if (renderer_ == NULL)
    throw std::runtime_error("Renderer could not be created! SDL_Error: " + std::string(SDL_GetError())); 

  if (TTF_Init() == -1)
    throw std::runtime_error("SDL_ttf could not initialize! TTF_Error: " + std::string(TTF_GetError()));

  font_ = TTF_OpenFont("resources/Winamp.ttf", FONT_SIZE);
  if (!font_)
    throw std::runtime_error("Failed to load font! TTF_Error: " + std::string(TTF_GetError()));

  if (SDL_Init(SDL_INIT_AUDIO) < 0)
    throw std::runtime_error("Failed to initialize SDL:" + std::string(SDL_GetError()));

  if (Mix_OpenAudio(sampleRate_, MIX_DEFAULT_FORMAT, channels_, 2048) < 0)
    throw std::runtime_error("Failed to initialize SDL_mixer: " + std::string(Mix_GetError()));

  Mix_VolumeMusic(MIX_MAX_VOLUME / 2);

  cleanWindowDraws();
  SDL_RenderPresent(renderer_);
}


DisplayWindow::~DisplayWindow() {
  SDL_DestroyRenderer(renderer_);
  SDL_DestroyWindow(window_);
  TTF_CloseFont(font_);
  Mix_CloseAudio();
  Mix_Quit();
  TTF_Quit();
  SDL_Quit();
}


void DisplayWindow::checkEvent() {
  SDL_Event e;
  if(SDL_PollEvent(&e) == 0) return;
  if (e.type == SDL_QUIT) exit(0);
}


void DisplayWindow::display(std::vector<float> data) {
  cleanWindowDraws();
  drawBarsGraph(data);
  drawInfoBox();
  SDL_RenderPresent(renderer_);
}


void DisplayWindow::cleanWindowDraws() {
  SDL_SetRenderDrawColor(
    renderer_,
    BG_RED,
    BG_GREEN,
    BG_BLUE,
    BG_ALPHA
  );
  SDL_RenderClear(renderer_);
}


void DisplayWindow::setBarColor(float dataPoint, int dataSize) {
  int overflowTimes = static_cast<int>(dataPoint / BARS_COLOR_CHANGE_OVERFLOW_VALUE);

  int red = BARS_BASE_RED;
  int green = BARS_BASE_GREEN;
  int blue = BARS_BASE_BLUE;
  int colorChange = BARS_BASE_COLOR_CHANGE_VALUE;

  while (overflowTimes > 0) {
    --overflowTimes;

    if ((red + colorChange) < 255) {
      red += colorChange;
      continue;
    } else {
      red = 255;
    }

    if ((green - colorChange) > 0) {
      green -= colorChange;
      continue;
    } else {
      green = 0;
    }

    if ((blue + colorChange) < 255) {
      blue += colorChange;
      continue;
    } else {
      blue = 255;
    }

    break; // All colors reached their target max.
  }

  SDL_SetRenderDrawColor(
    renderer_,
    red,
    green,
    blue,
    BARS_BASE_ALPHA
  );
}


void DisplayWindow::drawBarsGraph(std::vector<float> data) {
  int gap = 1;
  int xOffset = 5;
  int yOffset = SCREEN_HEIGHT - 5; 
  int barWidth = (SCREEN_WIDTH - xOffset - (data.size() * gap)) / data.size();
  barWidth = (barWidth <= 0) ? 1 : barWidth;

  for (int i = 0; i < data.size(); ++i) {
    int xPosition = static_cast<int>(xOffset + i * (barWidth + gap));
    int yPosition = yOffset;
    int barHeight = static_cast<int>(data[i]) % yOffset;
    SDL_Rect bar = { xPosition, yPosition, barWidth, -(barHeight) };
    setBarColor(data[i], data.size());
    SDL_RenderFillRect(renderer_, &bar);

    // Removed because bloated the display too much.
    // Draw bar's text
    //std::string dataPointStr = std::to_string(data[i]); // to_string always display 6 decimal of the float
    //dataPointStr.resize(dataPointStr.size() - 5); // This is a hack to truncate the string version
    //drawText(dataPointStr, xPosition, yPosition);
  }
}


void DisplayWindow::drawText(const std::string& text, int xPosition, int yPosition, SDL_Color textColor) {
  SDL_Surface* textSurface = TTF_RenderText_Solid(font_, text.c_str(), textColor);
  if (!textSurface) {
    std::cerr << "Unable to render text surface! TTF_Error: " << TTF_GetError() << std::endl;
    return;
  }

  SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer_, textSurface);
  if (!textTexture) {
    std::cerr << "Unable to create texture from rendered text! SDL_Error: " << SDL_GetError() << std::endl;
    SDL_FreeSurface(textSurface);
    return;
  }

  SDL_Rect renderQuad = {xPosition, yPosition, textSurface->w, textSurface->h};
  SDL_RenderCopy(renderer_, textTexture, NULL, &renderQuad);

  SDL_DestroyTexture(textTexture);
  SDL_FreeSurface(textSurface);
}


void DisplayWindow::drawInfoBox() {
  int boxWidth = 500;
  int boxHigth = 200;
  int boxXPosition = SCREEN_WIDTH - boxWidth - 10;
  int boxYPosition = 10; 

  // Draw the main box
  SDL_Rect box = {
    boxXPosition,
    boxYPosition,
    boxWidth,
    boxHigth
  };
  SDL_SetRenderDrawColor(
    renderer_,
    INFO_BOX_BG_RED,
    INFO_BOX_BG_GREEN,
    INFO_BOX_BG_BLUE,
    INFO_BOX_BG_ALPHA
  );
  SDL_RenderFillRect(renderer_, &box);

  // Draw the border
  SDL_Rect border = box;
  int borderWidth = 3;
  border.x -= borderWidth / 2;
  border.y -= borderWidth / 2;
  border.w += borderWidth;
  border.h += borderWidth;
  SDL_SetRenderDrawColor(
    renderer_,
    INFO_BOX_BORDER_RED,
    INFO_BOX_BORDER_GREEN,
    INFO_BOX_BORDER_BLUE,
    INFO_BOX_BORDER_ALPHA
  );
  SDL_RenderDrawRect(renderer_, &border);

  int textXPosition = boxXPosition + 1;
  int textYPosition = boxYPosition + 1;
  SDL_Color textColor = {
    INFO_BOX_TEXT_RED,
    INFO_BOX_TEXT_GREEN,
    INFO_BOX_TEXT_BLUE,
    INFO_BOX_TEXT_ALPHA
  };
  drawText(" SpectroWAVe ", textXPosition, textYPosition, textColor);
  drawText("+--------------------------+", textXPosition, textYPosition + (FONT_SIZE * 1), textColor);

  drawText("Currently playing: " + filepath_, textXPosition, textYPosition + (FONT_SIZE * 3), textColor);
  drawText("Sampling Rate: " + std::to_string(sampleRate_) + " Hz", textXPosition, textYPosition + (FONT_SIZE * 4), textColor);
  drawText("Channels: " + std::to_string(channels_), textXPosition, textYPosition + (FONT_SIZE * 5), textColor);
  drawText("Frames (per channel): " + std::to_string(frames_), textXPosition, textYPosition + (FONT_SIZE * 6), textColor);

  drawText("FFT Size: " + std::to_string(fftSize_), textXPosition, textYPosition + (FONT_SIZE * 8), textColor);
  drawText("Frequency Resolution: " + std::to_string(sampleRate_ / fftSize_) + " Hz", textXPosition, textYPosition + (FONT_SIZE * 9), textColor);
  drawText("Binning: " + std::to_string(binning_) + " channels", textXPosition, textYPosition + (FONT_SIZE * 10), textColor);
}


void DisplayWindow::startMusic() {
  std::cout << "Playing audio file at: " << filepath_.c_str() << std::endl;

  Mix_Music* music = Mix_LoadMUS(filepath_.c_str());
  if (!music) {
    std::cerr << "Failed to load sound file: " << Mix_GetError() << std::endl;
    return;
  }

  if (Mix_PlayMusic(music, 1) == -1) {
    std::cerr << "Failed to play music: " << Mix_GetError() << std::endl;
    return;
  }
}
