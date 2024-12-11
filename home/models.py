from django.db import models

class Product(models.Model):
    class_label = models.CharField(max_length=50)
    product_name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)
    category = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.product_name

class ProductPrice(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="prices")
    market_price = models.FloatField()
    price_date = models.DateField()

    class Meta:
        ordering = ['-price_date']

    def __str__(self):
        return f"{self.product.product_name} - {self.market_price} on {self.price_date}"
