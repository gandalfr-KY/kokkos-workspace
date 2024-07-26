#include <iostream>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

const int MAX_ITER = 50;

struct RGB {
    unsigned char r, g, b;
};

void saveBMP(const char* filename, Kokkos::View<RGB**, Kokkos::HostSpace> h_pixels) {
    const int HEIGHT = h_pixels.extent(0), WIDTH = h_pixels.extent(1);
    int filesize = 54 + 3 * WIDTH * HEIGHT;
    unsigned char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    unsigned char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(WIDTH    );
    bmpinfoheader[ 5] = (unsigned char)(WIDTH>> 8);
    bmpinfoheader[ 6] = (unsigned char)(WIDTH>>16);
    bmpinfoheader[ 7] = (unsigned char)(WIDTH>>24);
    bmpinfoheader[ 8] = (unsigned char)(HEIGHT    );
    bmpinfoheader[ 9] = (unsigned char)(HEIGHT>> 8);
    bmpinfoheader[10] = (unsigned char)(HEIGHT>>16);
    bmpinfoheader[11] = (unsigned char)(HEIGHT>>24);

    std::ofstream out(filename, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<char*>(bmpfileheader), 14);
    out.write(reinterpret_cast<char*>(bmpinfoheader), 40);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            RGB& pixel = h_pixels(HEIGHT - y - 1, x);
            unsigned char color[3] = {pixel.b, pixel.g, pixel.r};
            out.write(reinterpret_cast<char*>(color), 3);
        }
    }
    out.close();
}

Kokkos::View<RGB**, Kokkos::HostSpace> generateMandelbrot(const int HEIGHT, const int WIDTH) {
    Kokkos::View<RGB**, Kokkos::LayoutRight> pixels("pixels", HEIGHT, WIDTH);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {HEIGHT, WIDTH}), KOKKOS_LAMBDA(int y, int x) {
        Kokkos::complex<double> point((x - WIDTH/2.0) * 4.0 / WIDTH, (y - HEIGHT/2.0) * 4.0 / HEIGHT);
        Kokkos::complex<double> z(0, 0);
        int iter = 0;
        while (z.real() * z.real() + z.imag() + z.imag() < 4.0 && iter < MAX_ITER) {
            z = z*z + point;
            iter++;
        }
        if (iter < MAX_ITER) {
            pixels(y, x).r = iter;
            pixels(y, x).g = iter * 2;
            pixels(y, x).b = iter * 5;
        }
    });
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pixels);
}

int main() {
Kokkos::initialize();
{
    Kokkos::Timer tm;
    tm.reset();

    const int HEIGHT = 3200, WIDTH = 3200;
    auto h_pixels = generateMandelbrot(HEIGHT, WIDTH);

    std::cout << "duration: " << tm.seconds() << "\n";

    saveBMP("mandelbrot.bmp", h_pixels);

    std::cout << "Mandelbrot image saved as mandelbrot.bmp" << std::endl;

}
Kokkos::finalize();
}
