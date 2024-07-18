#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <iostream>
#include <fstream>
#include <sstream>


// Define the network
/**
 * @brief Define the neural network type.
 *
 * This network is a simple convolutional neural network (CNN) for multiclass classification.
 * The architecture includes:
 * - An input layer for 28x28 images.
 * - A convolutional layer with 16 filters of size 5x5 followed by ReLU activation.
 * - A max pooling layer with 2x2 pooling size.
 * - A fully connected layer with 120 neurons followed by ReLU activation.
 * - Another fully connected layer with 84 neurons followed by ReLU activation.
 * - A final fully connected layer with 10 neurons for the output classes.
 * - A loss layer for multiclass classification using logarithmic loss.
 */
using net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::relu<dlib::fc<84,
    dlib::relu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2,
    dlib::relu<dlib::con<16, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>;

/**
 * @brief Loads images and labels from a CSV file.
 *
 * This function reads a CSV file where each row represents an image in the form of pixel values
 * followed by the label. The CSV file format should have the label in the first column,
 * followed by 28*28 pixel values.
 *
 * @param filename The path to the CSV file.
 * @param images A vector to store the loaded images.
 * @param labels A vector to store the corresponding labels.
 */
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
