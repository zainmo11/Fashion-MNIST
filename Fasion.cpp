#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <iostream>
#include <fstream>
#include <sstream>


// Define the network
using net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::relu<dlib::fc<84,
    dlib::relu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2,
    dlib::relu<dlib::con<16, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>;

void load_csv(const std::string& filename, std::vector<dlib::matrix<unsigned char>>& images, std::vector<unsigned long>& labels) {
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<unsigned char> image(28 * 28);
        unsigned long label;

        // Get the label
        std::getline(ss, value, ',');
        label = std::stoul(value);

        // Get the image pixels
        for (size_t i = 0; i < 28 * 28; ++i) {
            std::getline(ss, value, ',');
            image[i] = static_cast<unsigned char>(std::stoul(value));
        }

        dlib::matrix<unsigned char> img;
        img.set_size(28, 28);
        for (size_t row = 0; row < 28; ++row) {
            for (size_t col = 0; col < 28; ++col) {
                img(row, col) = image[row * 28 + col];
            }
        }

        images.push_back(img);
        labels.push_back(label);
    }
}

int main() {
    // Load training data
    std::vector<dlib::matrix<unsigned char>> train_images;
    std::vector<unsigned long> train_labels;
    load_csv("fashion-mnist_train.csv", train_images, train_labels);

    // Load test data
    std::vector<dlib::matrix<unsigned char>> test_images;
    std::vector<unsigned long> test_labels;
    load_csv("fashion-mnist_test.csv", test_images, test_labels);

    // Define the neural network
    net_type net;
    dlib::dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.001);
    trainer.be_verbose();
    trainer.set_iterations_without_progress_threshold(350);

    // Enable CUDA
    trainer.set_synchronization_file("fashion_mnist_sync", std::chrono::seconds(100));
    trainer.set_mini_batch_size(128);

    // Train the network
    trainer.train(train_images, train_labels);

    // Evaluate on test set
    int correct = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        if (net(test_images[i]) == test_labels[i]) {
            ++correct;
        }
    }

    double accuracy = correct / static_cast<double>(test_images.size());
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Save the network to disk
    dlib::serialize("fashion_mnist_network.dat") << net;

    return 0;
}
