#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];//P矩阵
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];//V矩阵
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);
            auto current_position = frameInfo.m_position(x, y);
            int current_id = frameInfo.m_id(x, y);
            if(current_id == -1){
                continue;
            }

            Matrix4x4 currentModel = frameInfo.m_matrix[current_id];
            Matrix4x4 lastModel = m_preFrameInfo.m_matrix[current_id];
            Matrix4x4 currentModelInv = Inverse(currentModel);
            Float3 lastScreen = preWorldToScreen(
                preWorldToCamera(
                    currentModelInv(current_position, Float3::Point), Float3::Point
                ), Float3::Point
            );
            if(lastScreen.x <0 || lastScreen.x >= width || lastScreen.y < 0 || lastScreen.y >= height){
                continue;
            }
            int actualId = m_preFrameInfo.m_id(lastScreen.x, lastScreen.y);
            if(actualId == current_id){
                m_valid(x, y) = true;
                m_misc(x, y) = m_accColor(lastScreen.x, lastScreen.y);
            }

            // TODO: Reproject

        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            int x_min = std::max(0, x - kernelRadius);
            int x_max = std::min(width, x + kernelRadius);
            int y_min = std::max(0, y - kernelRadius);
            int y_max = std::min(height, y + kernelRadius);
            Float3 mean(0.0);
            for(int i = x_min; i < x_max; i++){
                for(int j = y_min; j < y_max; j++){
                    mean += curFilteredColor(i, j);
                }
            }
            mean /= std::max(static_cast<float>((x_max - x_min) * (y_max - y_min)), 0.0001f);
            Float3 variance(0.0);
            for(int i = x_min; i < x_max; i++){
                for(int j = y_min; j < y_max; j++){
                    variance += ((curFilteredColor(i, j) - mean) * (curFilteredColor(i, j) - mean));
                }
            }
            variance /= std::max(static_cast<float>((x_max - x_min) * (y_max - y_min)), 0.0001f);
            Float3 sigma = SafeSqrt(variance);
            Float3 color = m_accColor(x, y);
            Float3 color_last = Clamp(color, mean - sigma * m_colorBoxK, mean + sigma * m_colorBoxK);
            float alpha = 1.0f;
            if(m_valid(x, y)){
                alpha = m_alpha;
            }
            // TODO: Exponential moving average
            
            m_misc(x, y) = Lerp(color_last, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum_of_weight = 0.0;
            Float3 sum_of_weight_values(0.0);
            Float3 position_i = frameInfo.m_position(x, y);
            Float3 color_i = frameInfo.m_beauty(x, y);
            Float3 normal_i = frameInfo.m_normal(x, y);
            int boundary_x_max = std::min(kernelRadius + x, width);
            int boundary_x_min = std::max(0, x - kernelRadius);
            int boundary_y_max = std::min(kernelRadius + y, height);
            int boundary_y_min = std::max(y - kernelRadius, 0);
            for(int j_y = boundary_y_min; j_y < boundary_y_max; j_y++){
                for(int j_x = boundary_x_min; j_x < boundary_x_max; j_x++){
                    Float3 position_j = frameInfo.m_position(j_x, j_y);
                    Float3 color_j = frameInfo.m_beauty(j_x, j_y);
                    Float3 normal_j = frameInfo.m_normal(j_x, j_y);

                    //position
                    float p_w = - Dot(position_i - position_j, position_i - position_j) / (2.0 * m_sigmaCoord * m_sigmaCoord);

                    //color
                    float c_w = - Dot(color_i - color_j, color_i - color_j) / (2.0 * m_sigmaColor * m_sigmaColor);

                    //normal
                    float n_w = - SafeAcos(Dot(normal_i, normal_j)) * SafeAcos(Dot(normal_i, normal_j)) / (2.0 * m_sigmaNormal * m_sigmaNormal);
                    
                    //plane
                    float pl_w = - (Dot(normal_i,
                    (position_j - position_i) / std::max(SafeSqrt(Dot(position_j - position_i, position_j - position_i)), 0.0001f)
                    ) * Dot(normal_i,
                    (position_j - position_i) / std::max(SafeSqrt(Dot(position_j - position_i, position_j - position_i)), 0.0001f)
                    )) / (2.0 * m_sigmaPlane * m_sigmaPlane);

                    float weight = std::exp(p_w + c_w + n_w + pl_w);
                    sum_of_weight += weight;
                    sum_of_weight_values += (frameInfo.m_beauty(j_x, j_y) * weight);
                }
            }
            // TODO: Joint bilateral filter
            filteredImage(x, y) = sum_of_weight_values / std::max(sum_of_weight, 0.0001f);
        }
    }
    return filteredImage;
}

Buffer2D<Float3> Denoiser::Valelt_Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum_of_weight = 0.0;
            Float3 sum_of_weight_values(0.0);
            Float3 position_i = frameInfo.m_position(x, y);
            Float3 color_i = frameInfo.m_beauty(x, y);
            Float3 normal_i = frameInfo.m_normal(x, y);
            int boundary_x_max = std::min(kernelRadius + x, width);
            int boundary_x_min = std::max(0, x - kernelRadius);
            int boundary_y_max = std::min(kernelRadius + y, height);
            int boundary_y_min = std::max(y - kernelRadius, 0);
            for(int pass = 0; pass < 5; pass++){
                for(int i_left = -2; i_left <=2 ;i_left++){
                    for(int j_left = -2; j_left <= 2; j_left ++){
                        int j_x = x + i_left * std::pow(pass,2);
                        int j_y = y + j_left * std::pow(pass,2);
                        Float3 position_j = frameInfo.m_position(j_x, j_y);
                        Float3 color_j = frameInfo.m_beauty(j_x, j_y);
                        Float3 normal_j = frameInfo.m_normal(j_x, j_y);

                        //position
                        float p_w = - Dot(position_i - position_j, position_i - position_j) / (2.0 * m_sigmaCoord * m_sigmaCoord);

                        //color
                        float c_w = - Dot(color_i - color_j, color_i - color_j) / (2.0 * m_sigmaColor * m_sigmaColor);

                        //normal
                        float n_w = - SafeAcos(Dot(normal_i, normal_j)) * SafeAcos(Dot(normal_i, normal_j)) / (2.0 * m_sigmaNormal * m_sigmaNormal);
                        
                        //plane
                        float pl_w = - (Dot(normal_i,
                        (position_j - position_i) / std::max(SafeSqrt(Dot(position_j - position_i, position_j - position_i)), 0.0001f)
                        ) * Dot(normal_i,
                        (position_j - position_i) / std::max(SafeSqrt(Dot(position_j - position_i, position_j - position_i)), 0.0001f)
                        )) / (2.0 * m_sigmaPlane * m_sigmaPlane);

                        float weight = std::exp(p_w + c_w + n_w + pl_w);
                        sum_of_weight += weight;
                        sum_of_weight_values += (frameInfo.m_beauty(j_x, j_y) * weight);
                    }
                }
            }
            
            // TODO: Joint bilateral filter
            filteredImage(x, y) = sum_of_weight_values / std::max(sum_of_weight, 0.0001f);
        }
    }
    return filteredImage;
}
void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Valelt_Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
