package com.example.lab1_a;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import java.util.ArrayList;
import java.util.List;


public class Activity2 extends AppCompatActivity {

    EditText inputN;
    Button btnResult;
    TextView labelResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_2);

        this.inputN = this.findViewById(R.id.enter_n);
        this.btnResult = this.findViewById(R.id.btn_result);
        this.labelResult = this.findViewById(R.id.label_result);

        View.OnClickListener onButtonCountClick = r -> {

            String stringN = String.valueOf(inputN.getText());

            if (stringN.trim().equals("0")) {

                labelResult.setTextColor(getResources().getColor(R.color.error));
                labelResult.setText("Помилка: Введене число \nНЕ натуральне!");

            } else if (stringN.trim().equals("")) {

                labelResult.setTextColor(getResources().getColor(R.color.error));
                labelResult.setText("Помилка: Не введені дані!");

            } else {

                long n = Long.parseLong(stringN);

                Long[] multipliers = this.factors(n);

                StringBuilder result = new StringBuilder("Результат: n = ");

                for (int i = 0; i < multipliers.length - 1; i++) {
                    result.append(multipliers[i]).append(" * ");
                }

                result.append(multipliers[multipliers.length - 1]);

                if (multipliers[0] == n) {
                    result = new StringBuilder("Введене число просте.");
                }
                labelResult.setTextColor(getResources().getColor(R.color.black));
                labelResult.setText(result);

            }

        };

        btnResult.setOnClickListener(onButtonCountClick);
    }

    public Long[] factors(long n) {

        List<Long> multipliers = new ArrayList<>();

        while (n % 2 == 0) {

            multipliers.add(2L);
            n /= 2;

        }

        long[] sqrt = this.sumSqrt(n);
        multipliers.add(Math.abs(sqrt[0] + sqrt[1]));
        multipliers.add(Math.abs(sqrt[0] - sqrt[1]));

        return multipliers.toArray(new Long[0]);

    }

    public long[] sumSqrt(long n) {

        double x, y;

        x = Math.ceil(Math.sqrt(n));
        y = Math.pow(x, 2) - n;

        while (Math.sqrt(y) != Math.ceil(Math.sqrt(y))) {

            x++;
            y = Math.pow(x, 2) - n;

        }

        return new long[]{(long) x, (long) Math.sqrt(y)};

    }

}