package com.example.lab1_a;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.content.Intent;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    private Button btn_1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        this.btn_1 = (Button) this.findViewById(R.id.btn_1);

        btn_1.setOnClickListener(new Button.OnClickListener() {

            @Override
            public void onClick(View v) {

                Intent myIntent = new Intent(MainActivity.this, Activity2.class);
                MainActivity.this.startActivity(myIntent);
            }
        });
    }
}