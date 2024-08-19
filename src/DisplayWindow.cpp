#include "DisplayWindow.h"

DisplayWindow::DisplayWindow(FileSignalCollector* collector) :
  collector_(collector)
{
  if (!collector_->isLoaded())
    throw std::runtime_error("Collector must be loaded");

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

  if (Mix_OpenAudio(collector_->getSignalSampleRate(), MIX_DEFAULT_FORMAT, collector_->getSignalChannels(), 2048) < 0)
    throw std::runtime_error("Failed to initialize SDL_mixer: " + std::string(Mix_GetError()));

  Mix_VolumeMusic(MIX_MAX_VOLUME / 2);

  cleanWindowDraws();
  SDL_RenderPresent(renderer_);
}

DisplayWindow::~DisplayWindow() {
  // TODO
};

void DisplayWindow::checkEvent() {
  SDL_Event e;
  if(SDL_PollEvent(&e) == 0) return;
  if (e.type == SDL_QUIT) exit(0);
}

void DisplayWindow::display(std::vector<float> data) {
  cleanWindowDraws();
  drawInfoBox();
  drawBarsGraph(data);
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

void DisplayWindow::drawText(const std::string& text, int xPosition, int yPosition) {
  SDL_Color textColor = {255, 255, 255, 255};

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
  SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
  SDL_RenderFillRect(renderer_, &box);

  // Draw the border
  SDL_Rect border = box;
  int borderWidth = 3;
  border.x -= borderWidth / 2;
  border.y -= borderWidth / 2;
  border.w += borderWidth;
  border.h += borderWidth;
  SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
  SDL_RenderDrawRect(renderer_, &border);

  drawText("Currently playing: ", boxXPosition, boxYPosition);
  drawText("Currently playing: ", boxXPosition, boxYPosition + (FONT_SIZE * 1));
  drawText("Currently playing: ", boxXPosition, boxYPosition + (FONT_SIZE * 2));
  drawText("Currently playing: ", boxXPosition, boxYPosition + (FONT_SIZE * 3));
  drawText("Currently playing: ", boxXPosition, boxYPosition + (FONT_SIZE * 4));
}

void DisplayWindow::startMusic() {
  std::string filepath = collector_->getFilepath();

  std::cout << "Starting audio file at: " << filepath.c_str() << std::endl;
  Mix_Music* music = Mix_LoadMUS(filepath.c_str());
  if (!music) {
    std::cerr << "Failed to load sound file: " << Mix_GetError() << std::endl;
    return;
  }

  if (Mix_PlayMusic(music, 1) == -1) {
    std::cerr << "Failed to play music: " << Mix_GetError() << std::endl;
    return;
  }

}
