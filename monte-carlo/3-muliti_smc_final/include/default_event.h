#pragma once

#include "calibrated_parameters.h" 

// 违约事件结构体，用于稀疏化表示
struct DefaultEvent {
    int debtor_id;          // 违约债务人的ID
    int industry_id;        // 所属行业ID
    CreditRating rating;    // 信用评级
    double loss;            // 造成的损失
};