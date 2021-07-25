var business_policy_intensity = {
    'H7_Vaccination policy': 3,
    'C8_International travel controls': 3,
    'E1_Income support': 0,
    'H6_Facial Coverings': 3,
    'C5_Close public transport': 1,
    'H2_Testing policy': 3,
    'C4_Restrictions on gatherings': 3,
    'C2_Workplace closing': 2,
    'H3_Contact tracing': 2,
    'H8_Protection of elderly people': 2,
    'E2_Debt/contract relief': 0,
    'C7_Restrictions on internal movement': 2,
    'C6_Stay at home requirements': 2,
    'C3_Cancel public events': 2,
    'H1_Public information campaigns': 2,
    'C1_School closing': 2,
}

var mixed_policy_intensity = {
  'H7_Vaccination policy': 0,
  'C8_International travel controls': 0,
  'E1_Income support': 0,
  'H6_Facial Coverings': 0,
  'C5_Close public transport': 0,
  'H2_Testing policy': 0,
  'C4_Restrictions on gatherings': 0,
  'C2_Workplace closing': 0,
  'H3_Contact tracing': 0,
  'H8_Protection of elderly people': 0,
  'E2_Debt/contract relief': 0,
  'C7_Restrictions on internal movement': 0,
  'C6_Stay at home requirements': 0,
  'C3_Cancel public events': 0,
  'H1_Public information campaigns': 0,
  'C1_School closing': 0
}

var newest_real_contagion = 0.0130535210524699

var business_contagion_series = [
0.0131052211478249, 0.0131527601812281, 0.0131914871532075, 0.0132333553852817, 0.0132834982401704, 0.0133340012840648, 0.0133869517018042, 0.0134324079377706, 0.0134760209808605, 0.0135125236671314, 0.0135514318759391, 0.0135956261334372, 0.0136418846565079, 0.0136867153747602, 0.0137291229153647, 0.0137680209305558, 0.0138030465319474, 0.0138348659898924, 0.0138725093425905, 0.0139151973859129, 0.0139529276171828, 0.0139880863213051, 0.0140233460702888, 0.0140552352573735, 0.0140914947374853, 0.014125716974931, 0.0141660713348842, 0.0142082944154112, 0.0142453766717239, 0.0142785354552047
]

function postData(input) {
    $.ajax({
        type: "POST",
        url: url,
        data: {
          business_policy_intensity: business_policy_intensity,
          mixed_policy_intensity: mixed_policy_intensity,
          newest_real_contagion: newest_real_contagion,
          business_contagion_series: business_contagion_series
        },
        success: callbackFunc
    });
}

function callbackFunc(response) {
    // do something with the response
    console.log(response);
}

//postData('data to process');

function mixed_policies_forecast(business_policy_intensity, mixed_policy_intensity, newest_real_contagion, business_contagion_series) {
  // parepare offset values
  var policy_alphas = {   // intensity increase 1 unit, the contagion delta will multiply alpha
      'H7_Vaccination policy': 0.976,
      'C8_International travel controls': 0.979,
      'E1_Income support': 0.981,
      'H6_Facial Coverings': 0.982,
      'C5_Close public transport': 0.983,
      'H2_Testing policy': 0.984,
      'C4_Restrictions on gatherings': 0.985,
      'C2_Workplace closing': 0.986,
      'H3_Contact tracing': 0.988,
      'H8_Protection of elderly people': 0.989,
      'E2_Debt/contract relief': 0.991,
      'C7_Restrictions on internal movement': 0.991,
      'C6_Stay at home requirements': 0.993,
      'C3_Cancel public events': 0.995,
      'H1_Public information campaigns': 0.996,
      'C1_School closing': 0.998
  } // look up dictionary

  // prepare offset values
  var delta_y = []
  var pre_y = newest_real_contagion
  for (y of business_contagion_series){ // calculate for offset values
    //console.log(y)
    delta_y.push(y - pre_y)
    pre_y = y
  }

  console.log(delta_y)

  var policy_names = Object.keys(business_policy_intensity) //['policy1', 'policy2']

  //console.log(policy_names)

  for (const [key, policy] of Object.entries(policy_names)) {
    //console.log(policy)
    alpha = policy_alphas[policy]
    intensity_change = mixed_policy_intensity[policy] - business_policy_intensity[policy]

    // calculate for offset
    delta_cg = 1
    if (intensity_change < 0) {
      delta_cg = (1 / alpha) ** Math.abs(intensity_change)
    } else if (intensity_change > 0) {
      delta_cg = alpha ** intensity_change
    }

    //console.log( "delta_cg", delta_cg )
    //delta_y = [delta_cg * x for (x in delta_y)]
    var temp_dy = []
    for (const [a, x] of Object.entries(delta_y)) {
      //console.log(x)
      temp_dy.push(delta_cg * x);
    }
    delta_y = temp_dy
    //console.log(delta_y)
  }

  //console.log( delta_y )

  var mixed_policy_series = []
  pre_y = newest_real_contagion // rest again
  for (item of delta_y) {
    y = pre_y + item
    mixed_policy_series.push(y)
    pre_y = y
  }

  //console.log(business_contagion_series)

  return mixed_policy_series
}

mixed_policies_forecast(business_policy_intensity, mixed_policy_intensity, newest_real_contagion, business_contagion_series);