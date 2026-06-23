import pygame


class Renderer:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def draw(self, state, fps):
        self.screen.fill((0, 0, 0))

        # dibujar asteroides
        for pos in state.positions:
            pygame.draw.circle(
                self.screen,
                (200, 200, 200),
                (int(pos[0]), int(pos[1])),
                2
            )

        # mostrar FPS
        font = pygame.font.SysFont("Arial", 18)
        text = font.render(f"FPS: {int(fps)}", True, (0, 255, 0))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
