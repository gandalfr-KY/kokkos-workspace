#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <Kokkos_Core.hpp>

const int WIDTH = 12800;
const int HEIGHT = 12800;
const int MAX_ITER = 1000;

struct RGB {
    uint8_t r, g, b;
};

void saveBMP(const char* filename, Kokkos::View<RGB**> pixels) {
    int filesize = 54 + 3 * WIDTH * HEIGHT;
    uint8_t bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    uint8_t bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

    bmpfileheader[ 2] = (uint8_t)(filesize    );
    bmpfileheader[ 3] = (uint8_t)(filesize>> 8);
    bmpfileheader[ 4] = (uint8_t)(filesize>>16);
    bmpfileheader[ 5] = (uint8_t)(filesize>>24);

    bmpinfoheader[ 4] = (uint8_t)(WIDTH    );
    bmpinfoheader[ 5] = (uint8_t)(WIDTH>> 8);
    bmpinfoheader[ 6] = (uint8_t)(WIDTH>>16);
    bmpinfoheader[ 7] = (uint8_t)(WIDTH>>24);
    bmpinfoheader[ 8] = (uint8_t)(HEIGHT    );
    bmpinfoheader[ 9] = (uint8_t)(HEIGHT>> 8);
    bmpinfoheader[10] = (uint8_t)(HEIGHT>>16);
    bmpinfoheader[11] = (uint8_t)(HEIGHT>>24);

    auto pixels_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pixels);

    std::ofstream out(filename, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<char*>(bmpfileheader), 14);
    out.write(reinterpret_cast<char*>(bmpinfoheader), 40);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            RGB pixel = pixels_h(HEIGHT - y - 1, x);
            uint8_t color[3] = {pixel.b, pixel.g, pixel.r};
            out.write(reinterpret_cast<char*>(color), 3);
        }
    }
    out.close();
}

void generateMandelbrot(Kokkos::View<RGB**> pixels) {
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {HEIGHT, WIDTH}), KOKKOS_LAMBDA(int y, int x) {
        Kokkos::complex<double> point((x - WIDTH/2.0) * 4.0 / WIDTH, (y - HEIGHT/2.0) * 4.0 / HEIGHT);
        Kokkos::complex<double> z(0, 0);
        int iter = 0;
        while (z.real() * z.real() + z.imag() + z.imag() < 4.0 && iter < MAX_ITER) {
            z = z*z + point;
            iter++;
        }

        if (iter < MAX_ITER) {
            pixels(y, x).r = iter % 256;
            pixels(y, x).g = (iter * 2) % 256;
            pixels(y, x).b = (iter * 5) % 256;
        } else {
            pixels(y, x).r = 0;
            pixels(y, x).g = 0;
            pixels(y, x).b = 0;
        }
    });
}

int main() {
Kokkos::initialize();
{
    Kokkos::View<RGB**> pixels("state", HEIGHT, WIDTH);

    generateMandelbrot(pixels);
    saveBMP("mandelbrot.bmp", pixels);

    std::cout << "Mandelbrot image saved as mandelbrot.bmp" << std::endl;

}
Kokkos::finalize();
}
